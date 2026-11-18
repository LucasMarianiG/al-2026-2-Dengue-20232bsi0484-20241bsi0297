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
- **Vitória** (3205309): município escolhido pela equipe. Elegível: 294, 139, 48 e 93 semanas
  nos níveis 1 a 4 (mínimo exigido: 5) e 573 transições válidas (mínimo: 400); ver
  `results/tabelas/e1_segundo_municipio.csv`. Justificativa da escolha: a completar pela equipe.
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
src/               download.py, dados.py, markov.py, etapa1.py ... etapa8.py
tests/             test_markov.py, test_dados.py, test_etapa1.py, test_markov_e3.py, test_etapa3.py,
                   test_markov_e4.py, test_etapa4.py, test_markov_e5.py, test_etapa5.py
results/tabelas/   results/figuras/
relatorio/
```

## Estado do trabalho

| Etapa | Situação |
|---|---|
| E1 funções (`src/dados.py`), auditoria, elegibilidade, gráficos e testes | concluída |
| E1 conferência nos Canais 2 e 3 (10 semanas; boletim da Sesa) | pendente |
| E2 (matriz estimada, erros-padrão, Laplace) | pendente |
| E3 (π por três métodos, iterações, primitividade, Wielandt) | código, testes e resultados concluídos; revisão pendente |
| E4 (espectro, meia-vida, P^52 por duas vias, κ₁, curvas de decaimento) | código, testes e resultados concluídos; revisão pendente |
| E5 (permanência, histograma × geométrica, matriz M por dois caminhos, Kac, tempos empíricos) | código, testes e resultados concluídos; revisão pendente |
| E6 a E8 | pendentes |

## Figuras e tabelas do relatório → arquivo que as gera

Numeração no relatório a definir; `run_all.py` ainda não existe (gera tudo abaixo quando estiver pronto).

| Item | Arquivo | Gerado por |
|---|---|---|
| Auditoria dos 78 municípios (E1) | `results/e1_auditoria.csv` | `src/etapa1.py` |
| Escolha e elegibilidade do 2º município (E1) | `results/tabelas/e1_segundo_municipio.csv` | `src/etapa1.py` |
| Sequência de níveis de Serra (E1) | `results/figuras/e1_serie_serra.png` | `src/etapa1.py` |
| Mapa de calor município × ano (E1) | `results/figuras/e1_mapa_calor.png` | `src/etapa1.py` |
| π por método, iterações e resíduos (E3) | `results/tabelas/e3_pi_metodos.csv` | `src/etapa3.py` |
| Resumo da E3: menor k, Wielandt, \|λ₂\|, discordância (E3) | `results/tabelas/e3_resumo.csv` | `src/etapa3.py` |
| Convergência do método das potências em Serra (E3) | `results/figuras/e3_convergencia_serra.png` | `src/etapa3.py` |
| Autovalores (parte real, imaginária, módulo) por território (E4) | `results/tabelas/e4_espectro.csv` | `src/etapa4.py` |
| Resumo da E4: \|λ₂\|, meia-vida, κ₁(V), diferença de P^52, traço e determinante (E4) | `results/tabelas/e4_resumo.csv` | `src/etapa4.py` |
| Decaimento de \|\|e_i P^n − π\|\|₁ com a reta C\|λ₂\|^n, Serra, Vitória e ES (E4) | `results/figuras/e4_decaimento_serra.png`, `e4_decaimento_vitoria.png`, `e4_decaimento_es.png` | `src/etapa4.py` |
| Permanência do modelo × durações observadas sem censura (E5) | `results/tabelas/e5_permanencia.csv` | `src/etapa5.py` |
| Histograma do nível 4 × geométrica do modelo (E5) | `results/tabelas/e5_histograma_nivel4.csv`, `results/figuras/e5_hist_nivel4_serra.png`, `e5_hist_nivel4_vitoria.png`, `e5_hist_nivel4_es.png` | `src/etapa5.py` |
| Matriz M de primeira passagem por sistemas 3×3 e por Z (E5) | `results/tabelas/e5_M.csv` | `src/etapa5.py` |
| Resumo da E5: Kac, m14 e m41 do modelo × empíricos (E5) | `results/tabelas/e5_resumo.csv` | `src/etapa5.py` |

## Uso de IA

Declaração completa em `IA_REGISTRO.md` e no Apêndice A do relatório.