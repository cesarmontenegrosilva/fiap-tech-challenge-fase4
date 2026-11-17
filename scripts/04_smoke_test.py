"""Quick local retrieval test; does not depend on a running Ollama server."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voc.rag import answer
from src.voc.retriever import ReviewStore

if __name__ == "__main__":
    store = ReviewStore()
    questions = [
        "Quais problemas relacionados à entrega aparecem nos comentários?",
        "O que os clientes elogiam na experiência de compra?",
        "Qual é a cotação do bitcoin hoje?",
    ]
    for question in questions:
        result = answer(question, store, use_llm=False)
        print("\nPERGUNTA:", question)
        print("STATUS:", result["status"])
        for evidence in result["evidence"][:3]:
            print(f"[{evidence.code}] nota={evidence.review_score} sim={evidence.score:.3f} texto={evidence.text[:170]}")
