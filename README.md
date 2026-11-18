# Cadeias de Markov e a dinâmica dos alertas de dengue no Espírito Santo

Trabalho Prático A (Vigilância em Estados) — Álgebra Linear, IFES Campus Serra, 2026/2.
Professor: Fidelis Zanetti de Castro. Entrega: 08/12/2026, 23h59.

## Integrantes

| Nome | Matrícula | GitHub |
|---|---|---|
| Lucas Mariani Gomes | 20232bsi0484 | @LucasMarianiG |
| Caio Sperandio Santos | 20241bsi0297 | @sscaioss |

## Objetivo

Modelar o nível de alerta semanal de dengue (InfoDengue, 1 a 4) de municípios capixabas como
cadeia de Markov, estimar as matrizes de transição e usar Álgebra Linear para responder:
quanto tempo no vermelho, quanto tempo até o vermelho, por quanto tempo o presente informa o
futuro e se o modelo vale fora da amostra. O trabalho modela o comportamento de um indicador:
não estima casos, não prevê óbitos e não estabelece causas.

## Municípios analisados

- **Serra** (3205002): município comum a todas as equipes.
- **Vitória** (3205309): município escolhido pela equipe. Justificativa e elegibilidade
  (≥ 5 semanas em cada nível e ≥ 400 transições válidas) serão registradas após a Etapa 1.
- **ES agregado**: soma das matrizes de contagem dos 78 municípios.

## Ambiente

- Python 3.14.0 (≥ 3.10 exigido). Bibliotecas e versões em `requirements.txt`.

## Como instalar, executar e testar

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
python run_all.py
```

Tempo aproximado de `run_all.py`: a preencher quando existir. `run_all.py` usa somente a cópia
congelada dos dados, sem acessar a internet.

## Dados

Os dados do InfoDengue (2015 a 2025) e a lista de municípios do IBGE foram baixados uma única
vez por `src/download.py` e estão congelados em `data/raw/`. O arquivo `data/MANIFEST.txt`
registra data e hora, URL, número de linhas e SHA-256 de cada arquivo. O download não deve ser
executado de novo: o script recusa rodar se o manifesto já existe.

## Estrutura

```
README.md          IA_REGISTRO.md     requirements.txt     pytest.ini     run_all.py
data/raw/          data/MANIFEST.txt
src/               download.py, markov.py, etapa1.py ... etapa8.py
tests/             test_markov.py
results/tabelas/   results/figuras/
relatorio/
```

## Estado do trabalho

| Etapa | Situação |
|---|---|
| E1 funções (`src/dados.py`) e testes | concluída |
| E1 auditoria, gráficos, Canais 2 e 3 | em andamento |
| E2 a E8 | pendentes |

## Figuras e tabelas do relatório → arquivo que as gera

A preencher conforme as etapas forem concluídas.

## Uso de IA

Declaração completa em `IA_REGISTRO.md` e no Apêndice A do relatório.