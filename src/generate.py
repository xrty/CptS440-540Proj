"""
LLM generation module for the RAG Academic Paper QA System.

Formats retrieved passages into a structured prompt and sends it to the
Groq API (llama-3.3-70b-versatile) to generate a grounded, cited answer.

Placeholder behavior:
    If GROQ_API_KEY is empty (not configured), generate_answer() returns a
    clearly labeled placeholder string instead of raising an exception.
    This lets the pipeline run end-to-end for testing retrieval without needing
    a live API key.

Usage:
    from generate import Generator

    gen = Generator()
    answer = gen.generate_answer("What is attention?", top5_chunks)
    print(answer)
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    LOG_LEVEL,
)

logging.basicConfig(level=LOG_LEVEL, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Prompt template — instructs the LLM to answer only from provided context
# and to cite source papers by name and page number.
SYSTEM_PROMPT = """\
You are a precise research assistant for academic papers.
Answer the user's question using ONLY the provided context passages.
Do NOT use any outside knowledge or make assumptions beyond the given text.
After each fact or claim, include a citation in the format [Paper Title, p.N].
If the context does not contain enough information to answer the question,
respond with: "The provided passages do not contain sufficient information to answer this question."
Be concise and factual.
"""

USER_PROMPT_TEMPLATE = """\
Context passages:
{context}

Question: {question}

Answer (cite sources inline):
"""


class Generator:
    """
    Groq LLM wrapper for answer generation.

    Attributes:
        client: Groq API client instance, or None if no API key is configured.
        model:  Groq model name (from config).
    """

    def __init__(self) -> None:
        """
        Initialize the Groq client.

        If GROQ_API_KEY is empty, self.client is set to None and all
        generate_answer() calls return a placeholder string gracefully.
        """
        if not GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY is not set. Running in placeholder mode.\n"
                "  → Copy .env.example to .env and add your Groq API key."
            )
            self.client = None
        else:
            try:
                from groq import Groq
                self.client = Groq(api_key=GROQ_API_KEY)
                logger.info(f"Groq client initialized (model: {GROQ_MODEL})")
            except ImportError:
                logger.error("'groq' package not installed. Run: pip install groq")
                self.client = None

        self.model = GROQ_MODEL

    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format a list of chunk dicts into a numbered context block for the prompt.

        Each passage is prefixed with its index and citation metadata so the
        LLM knows which paper and page each passage comes from.

        Args:
            chunks: List of chunk dicts (must have "chunk_text", "source_paper",
                    "page_num"). Typically the top-5 after reranking.

        Returns:
            A multi-line string ready to be inserted into the prompt template.

        Example:
            >>> ctx = gen.format_context(top5_chunks)
            >>> print(ctx[:300])
        """
        parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("source_paper", "Unknown Paper")
            page = chunk.get("page_num", "?")
            text = chunk.get("chunk_text", "").strip()
            parts.append(f"[{i}] {title}, p.{page}\n{text}")
        return "\n\n".join(parts)

    def build_prompt(self, question: str, context: str) -> str:
        """
        Assemble the full user-turn prompt from question and formatted context.

        Args:
            question: The user's natural-language question.
            context:  Output of format_context().

        Returns:
            Formatted prompt string for the user turn of the conversation.

        Example:
            >>> prompt = gen.build_prompt("What is BERT?", context_str)
        """
        return USER_PROMPT_TEMPLATE.format(context=context, question=question)

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict],
        temperature: float = LLM_TEMPERATURE,
        max_tokens: int = LLM_MAX_TOKENS,
    ) -> str:
        """
        Generate a grounded answer given a question and retrieved passages.

        If the Groq client is not configured (empty API key), returns a
        placeholder string with the formatted context, so retrieval quality
        can still be inspected without a live API connection.

        Args:
            question:    The user's natural-language question.
            chunks:      List of chunk dicts to use as context (typically top-5).
            temperature: Sampling temperature (lower = more deterministic).
            max_tokens:  Maximum tokens in the generated response.

        Returns:
            Generated answer string. May include inline citations like
            [Attention Is All You Need, p.3].

        Example:
            >>> answer = gen.generate_answer("What is self-attention?", top5_chunks)
            >>> print(answer)
        """
        context = self.format_context(chunks)
        prompt = self.build_prompt(question, context)

        # ── Placeholder mode (no API key) ──────────────────────────────────────
        if self.client is None:
            return (
                "[PLACEHOLDER — Groq API key not configured]\n\n"
                "Retrieved context that would be sent to the LLM:\n\n"
                f"{context}\n\n"
                "Set GROQ_API_KEY in your .env file to enable answer generation."
            )

        # ── Live API call ──────────────────────────────────────────────────────
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content.strip()
            logger.debug(f"Generated answer ({len(answer)} chars)")
            return answer

        except Exception as exc:
            logger.error(f"Groq API call failed: {exc}")
            return f"[ERROR] LLM generation failed: {exc}"


def load_generator() -> Generator:
    """
    Convenience factory: construct and return a Generator instance.

    Returns:
        Initialized Generator.

    Example:
        >>> gen = load_generator()
        >>> answer = gen.generate_answer(question, top5_chunks)
    """
    return Generator()


# ── Quick self-test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mock_chunks = [
        {
            "chunk_id":     "1706.03762_p2_c0",
            "chunk_text":   (
                "An attention function can be described as mapping a query and a set "
                "of key-value pairs to an output, where the query, keys, values, and "
                "output are all vectors."
            ),
            "source_paper": "Attention Is All You Need",
            "arxiv_id":     "1706.03762",
            "page_num":     2,
            "file_path":    "",
            "score":        0.95,
        },
    ]

    gen = load_generator()
    answer = gen.generate_answer("What is an attention function?", mock_chunks)
    print("\n=== Generated Answer ===")
    print(answer)
