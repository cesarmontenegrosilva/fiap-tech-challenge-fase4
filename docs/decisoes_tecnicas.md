# Decisões técnicas e compromissos

## Embeddings e banco vetorial

**Opção:** Sentence Transformers, `intfloat/multilingual-e5-small`, embeddings normalizados e ChromaDB persistente. **Motivo:** funcionamento local, suporte multilíngue e consulta por metadados. Limitação: tempo de carga do modelo e limiar de similaridade não calibrado; validar com consultas rotuladas.

## Embeddings x frequência

Uma busca top-k identifica exemplos relevantes, não a prevalência estatística dos problemas. Em perguntas sobre o que é "mais frequente", os scripts contam menções temáticas por expressões regulares no corpus filtrado. Essas contagens não constituem verdade-terreno de classificação e podem falhar com ironia, negação ou sinônimos.

## Texto e metadados

Os documentos são avaliações, não pedidos inteiros, para maximizar a rastreabilidade. A categoria de um pedido com muitos itens é apenas representativa; não aplicar causalidade à relação entre nota, entrega e categoria. Comentários curtos (menos de 12 caracteres) são omitidos explicitamente.

## Estratégias de proteção

- Comentários recuperados são dados não confiáveis; o LLM recebe instruções de não segui-los.
- Cada evidência exibe identificador, nota, mês, categoria e comentário.
- Perguntas explicitamente fora do escopo são barradas; outras dependem do filtro de similaridade e do comportamento do modelo.
- O aplicativo mostra falha de Ollama sem fabricar uma resposta; ainda permite examinar evidências.

## Melhorias futuras (fora do escopo mínimo)

- Criar um conjunto ouro de consultas com relevâncias rotuladas, medir Recall@k/Precision@k de forma confiável.
- Classificador temático supervisionado validado, com tratamento de negações e tópicos múltiplos.
- Reranking, busca híbrida lexical + semântica, testes de segurança e confiança de citações por afirmação.
- Normalizar pedidos multitem, dimensões temporais e estimativas mais rigorosas.
