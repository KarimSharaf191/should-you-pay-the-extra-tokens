"""Model backends with one interface:

    generate(messages, thinking, max_new) -> (answer, reasoning, in_tokens, out_tokens)

``answer`` is the text after the thinking block and ``reasoning`` the thinking
trace, so both backends hand DSPy and the parsers the same kind of string.
"""

import gc
import os
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# Per-model request settings used for the paper's runs.
MODEL_SETTINGS = {
    # Qwen3-8B through the API: greedy, output capped per call, reasoning flag.
    "qwen/qwen3-8b": dict(backend="api", has_thinking=True, temperature=0,
                          send_max_tokens=True),
    # Qwen3-4B / 1.7B with Hugging Face Transformers: greedy with thinking off,
    # Qwen3's recommended sampling with thinking on.
    "Qwen/Qwen3-4B": dict(backend="hf", has_thinking=True),
    "Qwen/Qwen3-1.7B": dict(backend="hf", has_thinking=True),
    # Llama-3.2-3B-Instruct through the API: no thinking mode, no output cap.
    "meta-llama/llama-3.2-3b-instruct": dict(backend="api", has_thinking=False,
                                             temperature=0, top_p=1, seed=42,
                                             send_max_tokens=False),
}

HF_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
HF_SEED = 42
HF_CONTEXT = 32768


def free_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def split_think(raw):
    """(reasoning, answer) from raw text that may contain <think>...</think>."""
    if "</think>" in raw:
        reasoning, answer = raw.split("</think>", 1)
        return reasoning.replace("<think>", "").strip(), answer.strip()
    if "<think>" in raw:                       # never finished thinking
        return raw.replace("<think>", "").strip(), ""
    return "", raw.strip()


class APIBackend:
    """OpenRouter, or any OpenAI-compatible server (e.g. vLLM) via ``base_url``."""

    kind = "api"

    def __init__(self, model, base_url=OPENROUTER_URL, api_key=None,
                 temperature=0, top_p=None, seed=None, send_max_tokens=True,
                 max_retries=2, backoff=5, timeout=300, **_):
        self.name = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "EMPTY")
        self.temperature, self.top_p, self.seed = temperature, top_p, seed
        self.send_max_tokens = send_max_tokens
        self.max_retries, self.backoff, self.timeout = max_retries, backoff, timeout
        self.openrouter = "openrouter.ai" in self.base_url

    def generate(self, messages, thinking, max_new):
        body = {"model": self.name, "messages": messages, "temperature": self.temperature}
        if self.top_p is not None:
            body["top_p"] = self.top_p
        if self.seed is not None:
            body["seed"] = self.seed
        if self.send_max_tokens and max_new:
            body["max_tokens"] = max_new
        if thinking is not None:
            if self.openrouter:
                body["reasoning"] = {"enabled": thinking}
            else:   # vLLM: Qwen3's chat template switch
                body["chat_template_kwargs"] = {"enable_thinking": thinking}

        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(f"{self.base_url}/chat/completions",
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"},
                                     json=body, timeout=self.timeout)
                data = resp.json()
            except Exception as e:
                last_err = f"request failed: {e!r}"
                time.sleep(self.backoff * attempt)
                continue
            if "error" in data:
                last_err = f"API error: {data['error']}"
                code = data["error"].get("code") if isinstance(data["error"], dict) else None
                if code == 429:
                    retry_after = resp.headers.get("Retry-After")
                    time.sleep(float(retry_after) if retry_after else self.backoff * attempt * 3)
                else:
                    time.sleep(self.backoff * attempt)
                continue
            if not data.get("choices"):
                last_err = f"Unexpected response: {data}"
                time.sleep(self.backoff * attempt)
                continue
            msg = data["choices"][0].get("message", {})
            usage = data.get("usage", {}) or {}
            reasoning = msg.get("reasoning") or ""
            content = msg.get("content") or ""
            if not reasoning and "<think>" in content:    # vLLM without a reasoning parser
                reasoning, content = split_think(content)
            return (content.strip(), reasoning,
                    usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        raise RuntimeError(f"giving up after {self.max_retries} attempts: {last_err}")

    def truncate(self, document, max_new, max_chars=None):
        return document[:max_chars] if max_chars else document

    def close(self):
        pass


class HFBackend:
    """Local Hugging Face model with Qwen3's chat template."""

    kind = "hf"

    def __init__(self, model, sample_when_thinking=True, **_):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model
        self.sample_when_thinking = sample_when_thinking
        dtype = (torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                 else torch.float16)
        print(f"Loading {model} ({dtype}) ...")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, torch_dtype=dtype, device_map="auto")
        self.model.eval()

    def truncate(self, document, max_new, max_chars=None, overhead=1500):
        """Cut the document so the prompt fits the 32,768-token window."""
        if max_chars:
            document = document[:max_chars]
        budget = HF_CONTEXT - max_new - overhead
        ids = self.tok.encode(document, add_special_tokens=False)
        return document if len(ids) <= budget else self.tok.decode(ids[:budget])

    def generate(self, messages, thinking, max_new):
        torch = self.torch
        with torch.inference_mode():
            prompt = self.tok.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True,
                                                  enable_thinking=bool(thinking))
            inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
            n_in = int(inputs["input_ids"].shape[1])
            if n_in + max_new > HF_CONTEXT:
                raise ValueError(f"prompt too long for context window ({n_in} + {max_new} > {HF_CONTEXT})")
            if thinking and self.sample_when_thinking:
                torch.manual_seed(HF_SEED)
                gen = dict(HF_SAMPLING)
            else:
                gen = dict(do_sample=False)
            out = self.model.generate(**inputs, max_new_tokens=max_new,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id,
                                      **gen)
        new_ids = out[0, n_in:]
        # <think> tags are not special tokens in Qwen3, so they survive decoding.
        reasoning, answer = split_think(self.tok.decode(new_ids, skip_special_tokens=True))
        return answer, reasoning, n_in, int(new_ids.shape[0])

    def close(self):
        del self.model, self.tok
        free_gpu()


def make_backend(model, base_url=None, backend=None):
    settings = dict(MODEL_SETTINGS.get(model, {}))
    kind = backend or settings.pop("backend", "api")
    settings.pop("backend", None)
    settings.pop("has_thinking", None)
    if kind == "hf":
        return HFBackend(model, **settings)
    if base_url:
        settings["base_url"] = base_url
    return APIBackend(model, **settings)


def has_thinking(model):
    return MODEL_SETTINGS.get(model, {}).get("has_thinking", True)


# ------------------------------------------------------------------ DSPy
def dspy_lm(backend, thinking, max_new):
    """A DSPy LM that sends every call through ``backend`` (same code path for
    API and local models). Only the answer after the thinking block goes back
    to DSPy, so its field parser works."""
    import dspy
    from litellm.types.utils import ModelResponse, Choices, Message, Usage

    class BackendLM(dspy.BaseLM):
        def __init__(self):
            super().__init__(model=f"{backend.kind}/{backend.name}", model_type="chat",
                             temperature=0.0, max_tokens=max_new, cache=False)
            self.last_usage = (0, 0)
            self.last_text = ""

        def forward(self, prompt=None, messages=None, **kwargs):
            if messages is None:
                messages = [{"role": "user", "content": prompt}]
            answer, _r, in_tok, out_tok = backend.generate(messages, thinking, max_new)
            self.last_usage = (in_tok, out_tok)
            self.last_text = answer
            return ModelResponse(
                choices=[Choices(finish_reason="stop", index=0,
                                 message=Message(content=answer, role="assistant"))],
                usage=Usage(prompt_tokens=in_tok, completion_tokens=out_tok,
                            total_tokens=in_tok + out_tok),
                model=self.model)

    return BackendLM()
