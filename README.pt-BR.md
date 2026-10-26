# Laya Decision Lab

**Da pergunta à resposta fundamentada: fine-tuning, busca híbrida e IA generativa.**

Projeto pessoal de engenharia de IA de Everton Soares / GrowthTech, desenvolvido
com assistência de IA e validação humana. A documentação técnica principal está
em [inglês](README.md).

## O que construímos

- Fine-tuning completo de um modelo Laya multilíngue, com cerca de 322 milhões de
  parâmetros, em CPU.
- Preparação e revisão de perguntas reais, divisão dos dados e análise de erros.
- Base de fatos com fontes, restrições e controle de exportação para APIs.
- Busca híbrida: embeddings via OpenRouter, Qdrant local, BM25 e fusão de rankings.
- Respostas fechadas por regras e rascunhos abertos com Gemini Flash Lite.
- Pipeline em Python com limites de chamadas, relatórios e integração por worker.

## A distinção mais importante

O Laya identifica o **assunto**, não decide sozinho a resposta. O RAG recupera
evidências; o código confere o escopo; a IA generativa escreve quando necessário.
Encontrar informação parecida não é autorização para inventar experiência.

## Resultados e limites

O experimento escolhido usou 113 perguntas de treino, 12 de validação e 12 de
teste diagnóstico. Resultado: 10/12 no teste e 9/12 na validação. São conjuntos
pequenos e já inspecionados, não uma estimativa imparcial de qualidade em produção.
Leia o [histórico de treinamento](docs/TRAINING.md).

A integração com a automação LinkedIn existe em outro projeto. Esta pasta é o
motor de classificação, recuperação e respostas, não o navegador automatizado.
Problemas de paginação ou seletores não são resolvidos pelo fine-tuning.

## Demonstração pública offline

Com Python 3.12, sem chave de API e sem baixar modelo:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -X utf8 scripts/run_pipeline.py --public-demo
python -X utf8 -m unittest discover -s tests -v
```

Se já existe um ambiente virtual, apenas ative-o. A demo usa dados fictícios e
valida nove casos do resolvedor existente. **Não executa Laya nem RAG**: demonstra
o contrato de respostas e as recusas por falta de evidência. Foi verificada em
ambiente novo, sem dependências. O pipeline completo ainda depende dos artefatos
de treinamento e da base privada. CI Windows/Linux preparado, ainda não executado
no GitHub.

## Guias

- [Arquitetura e mapas](docs/ARCHITECTURE.md)
- [Embeddings, chunks, Qdrant e BM25](docs/RETRIEVAL.md)
- [Treinamento, pesos e overfitting](docs/TRAINING.md)
- [Histórico de construção](docs/ENGINEERING_JOURNAL.md)
- [Reprodução e dependências locais](docs/REPRODUCIBILITY.md)
- [Plano de vídeos e apresentação](docs/MEDIA.md)
- [Roadmap](docs/ROADMAP.md)

Os dois vídeos originais estão preservados localmente. Dados pessoais, chaves,
gravações brutas e checkpoints não serão publicados automaticamente. A V1 é
experimental; a reprodução do pipeline completo a partir de um clone ainda está em curso.
