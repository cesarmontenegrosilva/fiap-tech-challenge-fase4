"""Streamlit customer review intelligence interface."""
from __future__ import annotations

import streamlit as st

from src.voc.analysis import filter_reviews, load_reviews, topic_stats
from src.voc.config import COLLECTION_NAME, MIN_SIMILARITY, OLLAMA_MODEL, REVIEWS_CSV, TOP_K
from src.voc.rag import answer
from src.voc.retriever import ReviewStore

st.set_page_config(page_title="Voz do Cliente | Olist RAG", page_icon="🛒", layout="wide")
st.title("🛒 Voice of Customer Intelligence — Olist")
st.caption("Perguntas sobre avaliações reais • Recuperação vetorial • Evidências identificáveis")

if not REVIEWS_CSV.exists():
    st.error("A base ainda não foi preparada. Execute: python scripts/01_prepare.py")
    st.stop()

@st.cache_data(show_spinner=False)
def read_data():
    return load_reviews()

@st.cache_resource(show_spinner="Carregando embeddings e banco vetorial...")
def read_store():
    return ReviewStore()

df = read_data()
with st.sidebar:
    st.header("Filtros")
    choice = st.selectbox("Notas", ["Todas", "Notas 1 e 2", "Apenas nota 1", "Notas 1 a 3"])
    score_map = {"Todas": None, "Notas 1 e 2": 2, "Apenas nota 1": 1, "Notas 1 a 3": 3}
    score_max = score_map[choice]
    selected_year = st.selectbox("Ano da compra", ["Todos"] + sorted(
        [int(y) for y in df["year"].dropna().unique() if int(y) > 0]
    ))
    categories = ["Todas"] + df["category"].fillna("nao_informada").value_counts().head(35).index.tolist()
    category = st.selectbox("Categoria representativa", categories)
    late_only = st.checkbox("Somente entregas atrasadas (quando disponíveis)")
    top_k = st.slider("Avaliações recuperadas", 3, 15, TOP_K)
    min_similarity = st.slider("Similaridade mínima (heurística)", 0.50, 0.95, float(MIN_SIMILARITY), 0.01)
    use_llm = st.checkbox("Gerar resposta com Ollama", value=True)
    st.caption(f"Coleção: {COLLECTION_NAME} • Modelo LLM: {OLLAMA_MODEL}")
    st.caption("Sem dados suficientes, o sistema deve recusar conclusões. Filtros aplicam-se às buscas e às contagens.")

subset = filter_reviews(df, score_max, None if category == "Todas" else category,
                        None if selected_year == "Todos" else int(selected_year), late_only)
col1, col2, col3 = st.columns(3)
col1.metric("Avaliações com texto (filtro)", f"{len(subset):,}")
col2.metric("Notas baixas (1–2)", f"{int(subset['review_score'].le(2).sum()):,}")
col3.metric("Categorias disponíveis", f"{df['category'].nunique():,}")

st.subheader("Pergunte sobre as avaliações")
question = st.text_input(
    "Pergunta em português",
    placeholder="Quais reclamações sobre entrega são mencionadas por clientes com notas baixas?",
)
if st.button("Pesquisar e responder", type="primary", disabled=not question.strip()):
    try:
        store = read_store()
        with st.spinner("Buscando comentários e construindo resposta..."):
            outcome = answer(
                question, store, top_k=top_k, min_similarity=min_similarity,
                score_max=score_max, category=None if category == "Todas" else category,
                year=None if selected_year == "Todos" else int(selected_year),
                late_only=late_only, use_llm=use_llm,
            )
        st.session_state["last_result"] = outcome
        st.session_state["last_question"] = question
    except Exception as exc:
        st.error(f"Erro na consulta: {exc}")

if "last_result" in st.session_state:
    outcome = st.session_state["last_result"]
    st.subheader("Resposta")
    st.info(outcome["answer"])
    st.caption("Status: " + outcome["status"])
    if outcome["counts"] is not None:
        st.subheader("Contagens por palavras-chave na base filtrada")
        st.dataframe(outcome["counts"], use_container_width=True, hide_index=True)
        st.caption("Uma avaliação pode conter múltiplos temas. As contagens são heurísticas, não categorias validadas manualmente.")
    st.subheader("Evidências recuperadas")
    if not outcome["evidence"]:
        st.write("Nenhuma avaliação foi selecionada.")
    for ev in outcome["evidence"]:
        with st.expander(f"[{ev.code}] Nota {ev.review_score} · similaridade {ev.score:.3f} · {ev.purchase_month}"):
            st.write(ev.text)
            st.caption(f"review_id: {ev.review_id} | order_id: {ev.order_id} | categoria: {ev.category} | atraso: {ev.is_late}")

st.divider()
with st.expander("Exploração quantitativa opcional"):
    chart = topic_stats(subset)
    st.bar_chart(chart.set_index("tema")["avaliacoes"])
    st.caption("Menções aproximadas, contadas por expressões regulares na base filtrada.")

st.caption("Fonte: Brazilian E-Commerce Public Dataset by Olist (2016–2018). O sistema não acessa dados de clientes em tempo real.")
