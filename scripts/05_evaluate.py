"""Creates an auditable test table for human groundedness review.

Retrieval is unsupervised; recall@k requires gold relevant review IDs.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
from src.voc.config import ARTIFACTS
from src.voc.rag import answer
from src.voc.retriever import ReviewStore

QUESTIONS = [
    ("Quais clientes reclamam de atraso na entrega?", "entrega"),
    ("Quais avaliações mencionam produtos quebrados?", "qualidade"),
    ("Existem elogios ao atendimento?", "atendimento"),
    ("Que avaliações mencionam embalagem danificada?", "embalagem"),
    ("Qual é o preço do bitcoin hoje?", "fora_de_escopo"),
    ("Os clientes gostam do prazo de entrega?", "entrega"),
]
if __name__ == "__main__":
    store = ReviewStore()
    ARTIFACTS.mkdir(exist_ok=True)
    target = ARTIFACTS / "evaluation_manual.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["pergunta", "tema_esperado", "status",
                                                 "evidencias", "maior_similaridade",
                                                 "relevancia_manual_0_1", "fundamentacao_manual_0_1", "observacoes"])
        writer.writeheader()
        for question, topic in QUESTIONS:
            result = answer(question, store, use_llm=False)
            writer.writerow({"pergunta": question, "tema_esperado": topic,
                             "status": result["status"],
                             "evidencias": ";".join(x.review_id for x in result["evidence"]),
                             "maior_similaridade": max([x.score for x in result["evidence"]], default=""),
                             "relevancia_manual_0_1": "", "fundamentacao_manual_0_1": "", "observacoes": ""})
    print(f"Avaliação salva em {target}. Preencha os campos manuais após verificar as respostas.")
