"""Prompts and DSPy signatures, exactly as used for the paper's runs.

FinDVer: the instruction adapts the direct-answer prompt released with
FinDVer (Zhao et al., 2024). CoverBench: the zero-shot prompt is the one
released with CoverBench (Jacovi et al., 2024).
"""

import dspy

# ---------------------------------------------------------------- FinDVer
FINDVER_SYSTEM = """
As a financial expert, your task is to assess the truthfulness of the given statement
by determining whether it is entailed or refuted based on the provided financial document.

Definitions:
- entailed = TRUE
- refuted = FALSE
Output ONLY one word:

TRUE
or
FALSE
"""


def findver_user(document, statement):
    """User message for zero-shot (full report) and RAG (retrieved paragraphs)."""
    return f"""Financial Document:
{document}

Statement:
{statement}

Answer:"""


class FinDVerTyped(dspy.Signature):
    """As a financial expert, your task is to assess the truthfulness of the
    given statement by determining whether it is entailed or refuted based
    on the provided financial document. You should directly output the
    entailment label ('entailed' or 'refuted') without any intermediate
    steps."""

    document: str = dspy.InputField(desc="Financial Report")
    statement: str = dspy.InputField(desc="Claim to verify")
    answer: str = dspy.OutputField(
        desc="Directly output the entailment label of the claim: exactly one word, either 'entailed' or 'refuted'"
    )


class FinDVerCoT(dspy.Signature):
    """As a financial expert, assess the truthfulness of the statement by
    determining whether it is entailed or refuted based on the provided
    financial document. entailed = TRUE, refuted = FALSE."""

    document: str = dspy.InputField(desc="Full financial report text")
    statement: str = dspy.InputField(desc="Statement to verify")
    answer: str = dspy.OutputField(desc="Exactly one word: TRUE or FALSE")


FINDVER_AGENT_SYSTEM = """You are verifying a statement against a financial report.
You cannot see the full report. You can SEARCH for relevant excerpts by
issuing queries. On each turn, respond in ONE of these two formats only:

SEARCH: <a short query to find relevant report passages>
or
ANSWER: <one of: TRUE, FALSE>

TRUE means the statement is entailed by the report; FALSE means it is refuted.
Issue SEARCH queries until you have enough evidence, then give ANSWER.
You have a maximum of {max_steps} searches."""

FINDVER_AGENT = {
    "system": FINDVER_AGENT_SYSTEM,
    "opening": "Statement: {claim}\n\nBegin.",
    "continue": "Issue another SEARCH or give your ANSWER.",
    "forced": "You are out of searches. Reply with ANSWER: <TRUE|FALSE>",
}

# ------------------------------------------------------------- CoverBench
def coverbench_zeroshot(claim, context):
    """CoverBench paper's zero-shot prompt, sent as a single user message."""
    return (
        "Your task is to check if the Claim is correct according to the Evidence. "
        "Generate 'Correct' if the Claim is correct according to the Evidence, "
        "or 'Incorrect' if the claim is incorrect or cannot be verified.\n\n"
        f"Evidence: {context}\n"
        f"Claim: {claim}\n"
        "Answer:"
    )


# Used as the CoverBench RAG system prompt and as the CoT signature
# instruction; it asks for step-by-step reasoning before the verdict.
COVERBENCH_REASONING = (
    "Your task is to check if the Claim is correct according to the Evidence. "
    "First, think through your reasoning step by step, citing the relevant "
    "parts of the Evidence that support or contradict the Claim. "
    "Then give your final verdict. "
    "Generate 'Correct' if the Claim is correct according to the Evidence, or "
    "'Incorrect' if the claim is incorrect or cannot be verified."
)


def coverbench_rag_user(context, claim):
    return f"""Evidence:
{context}

Claim:
{claim}

Verdict:"""


class CoverBenchTyped(dspy.Signature):
    """Your task is to check if the Claim is correct according to the Evidence. Generate 'Correct' if the Claim is correct according to the Evidence, or 'Incorrect' if the claim is incorrect or cannot be verified."""

    evidence: str = dspy.InputField()
    claim: str = dspy.InputField()
    answer: str = dspy.OutputField(desc="'Correct' or 'Incorrect'")


class CoverBenchCoT(dspy.Signature):
    __doc__ = COVERBENCH_REASONING

    evidence: str = dspy.InputField()
    claim: str = dspy.InputField()
    verdict: str = dspy.OutputField(desc="Either 'Correct' or 'Incorrect', exactly one word")


COVERBENCH_AGENT_SYSTEM = """You are verifying a claim against its evidence.
You cannot see the full evidence. You can SEARCH for relevant excerpts by
issuing queries. On each turn, respond in ONE of these two formats only:

SEARCH: <a short query to find relevant evidence passages>
or
ANSWER: <one of: Correct, Incorrect>

Correct means the claim is correct according to the evidence; Incorrect means the
claim is incorrect or cannot be verified.
Issue SEARCH queries until you have enough evidence, then give ANSWER.
You have a maximum of {max_steps} searches."""

COVERBENCH_AGENT = {
    "system": COVERBENCH_AGENT_SYSTEM,
    "opening": "Claim: {claim}\n\nBegin.",
    "continue": "Issue another SEARCH or give your ANSWER.",
    "forced": "You are out of searches. Reply with ANSWER: <Correct|Incorrect>",
}
