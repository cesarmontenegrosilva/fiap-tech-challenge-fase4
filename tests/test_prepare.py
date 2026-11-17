import pandas as pd

from src.voc.prepare import clean_text, prepare


def test_clean_text_missing():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    assert clean_text("   Ótimo   produto\nembalado ") == "Ótimo produto embalado"


def test_prepare_keeps_text_and_enriches(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    pd.DataFrame([
        {"review_id": "r1", "order_id": "o1", "review_score": 1,
         "review_comment_title": "Horrível", "review_comment_message": "Chegou muito atrasado e quebrado!",
         "review_creation_date": "2018-02-10 12:00:00"},
        {"review_id": "r2", "order_id": "o2", "review_score": 5,
         "review_comment_title": "", "review_comment_message": None,
         "review_creation_date": "2018-02-10 12:00:00"},
    ]).to_csv(raw / "olist_order_reviews_dataset.csv", index=False)
    pd.DataFrame([
        {"order_id": "o1", "order_purchase_timestamp": "2018-01-01 10:00:00",
         "order_delivered_customer_date": "2018-01-10 10:00:00",
         "order_estimated_delivery_date": "2018-01-06 10:00:00"},
    ]).to_csv(raw / "olist_orders_dataset.csv", index=False)
    pd.DataFrame([{"order_id": "o1", "product_id": "p1"}]).to_csv(
        raw / "olist_order_items_dataset.csv", index=False)
    pd.DataFrame([{"product_id": "p1", "product_category_name": "beleza_saude"}]).to_csv(
        raw / "olist_products_dataset.csv", index=False)
    out = tmp_path / "processed.csv"
    summary = prepare(raw, out)
    result = pd.read_csv(out)
    assert summary["usable_text_reviews"] == 1
    assert result.loc[0, "delivery_delay_days"] == 4
    assert result.loc[0, "category"] == "beleza_saude"
    assert bool(result.loc[0, "is_late"])
    assert result.loc[0, "purchase_month"] == "2018-01"
