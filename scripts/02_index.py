"""Index the prepared Olist reviews into local persistent ChromaDB."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
from src.voc.retriever import ReviewStore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true", help="Apaga e reconstrói o índice")
    parser.add_argument("--limit", type=int, default=0, help="Indexar apenas N comentários para testes")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    store = ReviewStore()
    print("Total indexado:", store.index(args.rebuild, args.batch_size, args.limit))
