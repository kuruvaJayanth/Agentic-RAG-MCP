import json
import re


def refine_query(query: str) -> str:
    """
    Improve a user's question for semantic document retrieval.
    """

    query = query.strip()

    # Remove unnecessary whitespace
    query = re.sub(r"\s+", " ", query)

    # Add retrieval-oriented wording
    refined_query = (
        f"{query}. "
        "Find the most relevant information from the research documents."
    )

    return refined_query


def extract_entities(query: str) -> dict:
    """
    Extract important terms from the user's question.
    """

    words = query.split()

    # Simple keyword extraction for now.
    # We'll improve this later using Gemini.
    keywords = []

    stop_words = {
        "what",
        "why",
        "how",
        "when",
        "where",
        "which",
        "the",
        "is",
        "are",
        "of",
        "in",
        "on",
        "for",
        "and",
        "to",
        "a",
        "an"
    }

    for word in words:

        cleaned = word.strip(".,?!:;()[]{}")

        if (
            cleaned
            and cleaned.lower() not in stop_words
            and len(cleaned) > 2
        ):
            keywords.append(cleaned)

    return {
        "entities": list(dict.fromkeys(keywords))
    }


def check_relevance(
    query: str,
    retrieved_text: str
) -> dict:
    """
    Basic relevance check between the question and retrieved context.
    """

    query_words = set(
        query.lower().split()
    )

    context_words = set(
        retrieved_text.lower().split()
    )

    if not query_words:
        return {
            "relevant": False,
            "score": 0
        }

    overlap = query_words.intersection(
        context_words
    )

    score = len(overlap) / len(query_words)

    return {
        "relevant": score >= 0.2,
        "score": round(score, 3)
    }