"""Exploratory report of processed Olist comments; requires no extra plotting dependencies."""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.voc.analysis import load_reviews, topic_stats
from src.voc.config import ARTIFACTS

if __name__ == "__main__":
    df = load_reviews()
    ARTIFACTS.mkdir(exist_ok=True, parents=True)
    topics = topic_stats(df)
    topics.to_csv(ARTIFACTS / "topic_mention_counts.csv", index=False)
    df.groupby("review_score").size().rename("reviews").to_csv(ARTIFACTS / "scores.csv")
    df.groupby("purchase_month").size().rename("reviews").to_csv(ARTIFACTS / "reviews_by_month.csv")
    with (ARTIFACTS / "eda_report.md").open("w", encoding="utf8") as f:
        f.write("# Análise exploratória — Olist\n\n")
        f.write(f"- Comentários textuais válidos: **{len(df):,}**\n")
        f.write(f"- Avaliações de nota 1–2: **{int(df['review_score'].le(2).sum()):,}**\n")
        f.write(f"- Pedidos com atraso informado: **{int(df['is_late'].astype(str).str.lower().eq('true').sum()):,}**\n\n")
        f.write("## Menções por palavras-chave (categorias sobrepostas; não são rótulos de verdade)\n\n")
        f.write("| Tema | Avaliações com menção | Percentual da amostra (%) |\n")
        f.write("|---|---:|---:|\n")
        for record in topics.to_dict(orient="records"):
            f.write(f"| {record['tema']} | {record['avaliacoes']} | {record['percentual']:.2f} |\n")
        f.write("\nEssas contagens não equivalem a classificação manual ou inferência causal.\n")
    print("Gerado relatório em artifacts/eda_report.md e CSVs de apoio")
