"""Transparent keyword-based topic counting over all eligible reviews.

Note: these are keyword mentions, not an NLP ground truth or causal effect.
"""
from __future__ import annotations

import re
import pandas as pd

from .config import REVIEWS_CSV

TOPIC_PATTERNS = {
    "Entrega e atraso": r"\b(?:entrega|entregou|entregue|atrasad[oa]s?|atrasou|prazo|transportadora|frete|chegou|chegada)\b",
    "Qualidade e defeitos": r"\b(?:qualidade|defeito|defeituos[oa]|quebrad[oa]|danificad[oa]|avariad[oa]|funciona|funcionou|estragad[oa])\b",
    "Atendimento": r"\b(?:atendimento|atendente|suporte|resposta|contato|sac)\b",
    "Preço e pagamento": r"\b(?:preço|preco|caro|barato|valor|pagamento|cobrança|cobranca|paguei)\b",
    "Embalagem": r"\b(?:embalagem|embalado|embrulhado|caixa|pacote)\b",
    "Troca e devolução": r"\b(?:troca|trocar|devolução|devolucao|devolver|reembolso|estorno)\b",
}


def filter_reviews(df: pd.DataFrame, score_max: int | None = None,
                   category: str | None = None, year: int | None = None,
                   late_only: bool = False) -> pd.DataFrame:
    result = df
    if score_max is not None:
        result = result[result["review_score"] <= score_max]
    if category and category != "Todas":
        result = result[result["category"] == category]
    if year:
        result = result[result["year"] == year]
    if late_only:
        result = result[result["is_late"].astype(str).str.lower() == "true"]
    return result


def topic_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Review-level counts; themes can overlap, so percentages do not sum to 100%."""
    texts = df["text"].fillna("").astype(str)
    n = len(df)
    rows = []
    for topic, pattern in TOPIC_PATTERNS.items():
        count = int(texts.str.contains(pattern, regex=True, flags=re.IGNORECASE).sum())
        rows.append({"tema": topic, "avaliacoes": count,
                     "percentual": round(100 * count / n, 2) if n else 0})
    return pd.DataFrame(rows).sort_values("avaliacoes", ascending=False).reset_index(drop=True)


def load_reviews() -> pd.DataFrame:
    if not REVIEWS_CSV.exists():
        raise FileNotFoundError("Execute python scripts/01_prepare.py")
    return pd.read_csv(REVIEWS_CSV)
