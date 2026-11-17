from src.voc.rag import answer, build_prompt
from src.voc.retriever import Evidence


class DummyStore:
    def search(self, *args, **kwargs):
        return []


def test_out_of_scope_without_retrieval():
    result = answer("Qual é o preço do bitcoin?", DummyStore(), use_llm=False)
    assert result["status"] == "out_of_scope"


def test_insufficient_evidence():
    result = answer("Avaliações sobre água em Marte", DummyStore(), use_llm=False)
    assert result["status"] == "insufficient"


def test_prompt_cites_source():
    ev = Evidence("E1", "doc", "O produto chegou atrasado", 0.92, 1,
                  "r1", "o1", "livros", "2018-04", True)
    prompt = build_prompt("Houve atraso?", [ev])
    assert "[E1]" in prompt
    assert "O produto chegou atrasado" in prompt
