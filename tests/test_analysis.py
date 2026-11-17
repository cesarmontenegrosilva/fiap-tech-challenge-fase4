import pandas as pd
from src.voc.analysis import filter_reviews, topic_stats


def test_count_and_filter():
    df = pd.DataFrame([
        {"text": "Entrega chegou muito atrasada", "review_score": 1,
         "category": "livros", "year": 2018, "is_late": True},
        {"text": "Produto excelente, bom preço", "review_score": 5,
         "category": "livros", "year": 2018, "is_late": False},
        {"text": "Atendimento ruim e produto quebrado", "review_score": 1,
         "category": "saude", "year": 2017, "is_late": False},
    ])
    small = filter_reviews(df, score_max=2, category="livros")
    assert len(small) == 1
    stats = topic_stats(df).set_index("tema")
    assert stats.loc["Entrega e atraso", "avaliacoes"] == 1
    assert stats.loc["Qualidade e defeitos", "avaliacoes"] == 1
