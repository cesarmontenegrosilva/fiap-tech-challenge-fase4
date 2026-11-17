"""Deterministic cleaning and enrichment of the Olist order reviews dataset."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from .config import RAW, REVIEWS_CSV


REQUIRED_REVIEW_FIELDS = {
    "review_id", "order_id", "review_score", "review_comment_message"
}


def clean_text(value: object) -> str:
    """Normalize whitespace; missing reviews become empty strings."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def read_required(path: Path, fields: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo obrigatório não encontrado: {path}. Baixe e extraia a Olist em data/raw/."
        )
    df = pd.read_csv(path, low_memory=False)
    missing = fields - set(df.columns)
    if missing:
        raise ValueError(f"O CSV {path.name} não possui as colunas: {sorted(missing)}")
    return df


def build_categories(raw: Path) -> pd.DataFrame | None:
    items_file = raw / "olist_order_items_dataset.csv"
    products_file = raw / "olist_products_dataset.csv"
    if not (items_file.exists() and products_file.exists()):
        return None
    items = read_required(items_file, {"order_id", "product_id"})
    products = read_required(products_file, {"product_id", "product_category_name"})
    joined = items[["order_id", "product_id"]].drop_duplicates().merge(
        products[["product_id", "product_category_name"]],
        on="product_id", how="left", validate="many_to_one"
    )
    joined["product_category_name"] = joined["product_category_name"].fillna("nao_informada")
    # For multi-category orders use a deterministic representative category.
    # See README limitations; this is not a claim about every item in the order.
    categories = (joined.groupby("order_id")["product_category_name"]
                  .agg(lambda values: sorted(set(values.astype(str)))[0])
                  .reset_index(name="category"))
    return categories


def prepare(raw_dir: Path = RAW, output_file: Path = REVIEWS_CSV) -> dict:
    reviews = read_required(raw_dir / "olist_order_reviews_dataset.csv", REQUIRED_REVIEW_FIELDS)
    original_rows = len(reviews)

    if "review_comment_title" not in reviews:
        reviews["review_comment_title"] = ""
    reviews["text"] = reviews["review_comment_title"].map(clean_text)
    messages = reviews["review_comment_message"].map(clean_text)
    reviews["text"] = (reviews["text"] + " " + messages).str.strip()
    # Empty/short comments cannot contribute meaningful semantic evidence.
    reviews = reviews[reviews["text"].str.len() >= 12].copy()
    reviews["review_score"] = pd.to_numeric(reviews["review_score"], errors="coerce")
    reviews = reviews[reviews["review_score"].between(1, 5)].copy()
    reviews["review_score"] = reviews["review_score"].astype(int)
    reviews = reviews.drop_duplicates(subset=["review_id", "order_id", "text"])

    # Review creation month is always available even without the orders CSV.
    dt = pd.to_datetime(reviews.get("review_creation_date"), errors="coerce")
    if isinstance(dt, pd.Series):
        reviews["purchase_month"] = dt.dt.strftime("%Y-%m")
    else:
        reviews["purchase_month"] = "nao_informado"
    reviews["delivery_delay_days"] = -9999  # sentinel: information unavailable
    reviews["is_late"] = False

    orders_file = raw_dir / "olist_orders_dataset.csv"
    if orders_file.exists():
        orders = read_required(orders_file, {
            "order_id", "order_purchase_timestamp", "order_delivered_customer_date",
            "order_estimated_delivery_date"
        })
        orders = orders.drop_duplicates("order_id")
        for col in ["order_purchase_timestamp", "order_delivered_customer_date",
                    "order_estimated_delivery_date"]:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")
        reviews = reviews.merge(orders[["order_id", "order_purchase_timestamp",
                                         "order_delivered_customer_date",
                                         "order_estimated_delivery_date"]],
                                how="left", on="order_id", validate="many_to_one")
        purchased = reviews["order_purchase_timestamp"].dt.strftime("%Y-%m")
        reviews["purchase_month"] = purchased.fillna(reviews["purchase_month"])
        valid = reviews["order_delivered_customer_date"].notna() & \
            reviews["order_estimated_delivery_date"].notna()
        delay = ((reviews["order_delivered_customer_date"] -
                  reviews["order_estimated_delivery_date"]).dt.total_seconds() / 86400).round()
        reviews.loc[valid, "delivery_delay_days"] = delay[valid].astype(int)
        reviews["is_late"] = reviews["delivery_delay_days"] > 0

    cat = build_categories(raw_dir)
    if cat is not None:
        reviews = reviews.merge(cat, how="left", on="order_id", validate="many_to_one")
    else:
        reviews["category"] = "nao_informada"

    reviews["category"] = reviews["category"].fillna("nao_informada").astype(str)
    reviews["purchase_month"] = reviews["purchase_month"].fillna("nao_informado")
    reviews["year"] = pd.to_numeric(reviews["purchase_month"].str[:4], errors="coerce").fillna(0).astype(int)
    # Stable unique ID even if the source has repeated review_id values.
    reviews["document_id"] = reviews.apply(
        lambda row: hashlib.sha256(f"{row['review_id']}|{row['order_id']}|{row['text']}".encode("utf-8")).hexdigest()[:24],
        axis=1,
    )
    reviews = reviews.drop_duplicates("document_id")
    columns = ["document_id", "review_id", "order_id", "text", "review_score",
               "purchase_month", "year", "category", "delivery_delay_days", "is_late"]
    output_file.parent.mkdir(parents=True, exist_ok=True)
    reviews[columns].to_csv(output_file, index=False, encoding="utf-8")
    return {
        "source_rows": original_rows,
        "usable_text_reviews": len(reviews),
        "excluded_rows": original_rows - len(reviews),
        "output": str(output_file),
        "optional_orders": orders_file.exists(),
        "optional_categories": cat is not None,
    }
