"""Scaffold evaluation on FinDVer and CoverBench.

One pipeline for both benchmarks: full-document scaffolds (zero-shot, typed
DSPy prompt, DSPy chain-of-thought) and retrieval scaffolds (RAG, agentic RAG),
for API models (OpenRouter or any OpenAI-compatible server such as vLLM) and
local Hugging Face models. Run it with ``python -m src.findver_coverbench.run``.
"""
