from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voc.config import RAW

FILES = [
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_products_dataset.csv",
]
if __name__ == "__main__":
    for filename in FILES:
        path = RAW / filename
        print(("OK " if path.exists() else "AUSENTE ") + str(path))
    if not (RAW / FILES[0]).exists():
        raise SystemExit("ERRO: falta o CSV de avaliações. Veja README (download).")
