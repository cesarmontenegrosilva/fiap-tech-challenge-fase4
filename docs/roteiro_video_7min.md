# Roteiro de demonstração — vídeo de até 7 minutos

**Duração-alvo: 6min40s**, com 20s de margem para o limite.

| Tempo | O que mostrar | Mensagem |
|---|---|---|
| 0:00–0:35 | Capa do projeto / app | Problema: milhares de avaliações textuais pouco exploradas. |
| 0:35–1:20 | Diagrama Mermaid do README | Fonte Olist → preparação → embeddings → ChromaDB → recuperação → Ollama → evidências. |
| 1:20–2:05 | `scripts/01_prepare.py`, relatório EDA | Textos ausentes, notas, datas e categoria; limitações e deduplicação. |
| 2:05–2:50 | `scripts/02_index.py` e `retriever.py` | Embeddings E5 multilíngue, distância cosseno e metadados. |
| 2:50–4:20 | Streamlit, pergunta sobre entrega, filtro notas 1–2 | Exibir resposta, IDs [E1] etc. Abrir avaliações e conferir o texto. |
| 4:20–5:10 | Pergunta de tema frequente | Contagens por palavras-chave + exemplos; evitar afirmar frequências com top-k somente. |
| 5:10–5:45 | Pergunta sobre bitcoin ou assunto externo | Demonstrar recusa e tratamento de ausência de evidências. |
| 5:45–6:20 | Testes e `artifacts/evaluation_manual.csv` | Verificação de relevância, rastreabilidade, limitações. |
| 6:20–6:40 | Página do GitHub e README | Reprodutibilidade, dependências e execução local. |

## Recomendações de gravação

- Ajuste a fonte do editor/terminal, pois o vídeo será de curta duração.
- Antes da gravação, baixe o modelo Ollama e execute pelo menos uma consulta para carregá-lo na RAM.
- Tenha 3 perguntas preparadas, confirme que há evidências em cada filtro selecionado.
- Não afirme resultados quantitativos que não verificou nos dados; as contagens são aproximações por palavras-chave.
- No fim, mostre a URL do GitHub. Grave o vídeo com a ferramenta de sua preferência e entregue o link pelo canal definido pela instituição.
