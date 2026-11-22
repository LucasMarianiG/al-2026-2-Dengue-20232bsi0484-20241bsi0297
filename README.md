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
data/raw/          data/MANIFEST.txt   data/canal2_niveis_site.csv   data/canal3_boletim_sesa_2024.csv
src/               download.py, dados.py, markov.py, etapa1.py ... etapa8.py, canal2.py, canal3.py
tests/             test_markov.py, test_dados.py, test_etapa1.py, test_markov_e3.py, test_etapa3.py,
                   test_markov_e4.py, test_etapa4.py, test_markov_e5.py, test_etapa5.py,
                   test_canal2.py, test_canal3.py
results/tabelas/   results/figuras/
relatorio/
```

## Estado do trabalho

| Etapa | Situação |
|---|---|
| E1 funções (`src/dados.py`), auditoria, elegibilidade, gráficos e testes | concluída |
| E1 conferência nos Canais 2 e 3 (10 semanas por município; boletim da Sesa) | código, testes e resultados concluídos (ver abaixo); revisão pendente |
| E2 (matriz estimada, erros-padrão, Laplace) | pendente |
| E3 (π por três métodos, iterações, primitividade, Wielandt) | código, testes e resultados concluídos; revisão pendente |
| E4 (espectro, meia-vida, P^52 por duas vias, κ₁, curvas de decaimento) | código, testes e resultados concluídos; revisão pendente |
| E5 (permanência, histograma × geométrica, matriz M por dois caminhos, Kac, tempos empíricos) | código, testes e resultados concluídos; revisão pendente |
| E6 a E8 | pendentes |

## Conferências externas (Canais 2 e 3)

Esta é a parte do trabalho que depende de leitura manual, e por isso as leituras estão versionadas.

- **Canal 2** (`src/canal2.py`): 10 semanas sorteadas **para cada município** (resposta do professor em
  22/11/2026), com `numpy.random.default_rng(2026)`, Serra primeiro e depois Vitória, entre as semanas
  com nível registrado. O nível foi lido à mão na página de relatório de cada semana do InfoDengue
  (`info.dengue.mat.br/report/ES/<geocode>/<AAAASS>`) em 22/11/2026 e está em
  `data/canal2_niveis_site.csv` (cor do site: verde = 1, amarelo = 2, laranja = 3, vermelho = 4).
  Resultado: 20 de 20 semanas coincidem com os dados congelados. Divergências futuras, se houver,
  seriam explicadas por revisões retroativas e versão do modelo.
- **Canal 3** (`src/canal3.py`): ano escolhido, 2024. O 52º boletim da dengue da Sesa-ES não traz total
  anual por município (só o total do ES e a incidência de 4 semanas); os valores lidos do PDF, com o
  hash do arquivo, estão em `data/canal3_boletim_sesa_2024.csv`. Comparamos o total do ES (soma dos 78
  municípios na API) e a incidência de Serra nas SE 49 a 52; o total anual de Serra só existe na API.
  A coluna `casconf` da API vem zerada e `casprov` fica 31% abaixo do boletim: resultados mantidos e
  discutidos no relatório, sem escolher a coluna que "fecha melhor".

## Figuras e tabelas do relatório → arquivo que as gera

Numeração no relatório a definir; `run_all.py` ainda não existe (gera tudo abaixo quando estiver pronto).

| Item | Arquivo | Gerado por |
|---|---|---|
| Auditoria dos 78 municípios (E1) | `results/e1_auditoria.csv` | `src/etapa1.py` |
| Escolha e elegibilidade do 2º município (E1) | `results/tabelas/e1_segundo_municipio.csv` | `src/etapa1.py` |
| Sequência de níveis de Serra (E1) | `results/figuras/e1_serie_serra.png` | `src/etapa1.py` |
| Mapa de calor município × ano (E1) | `results/figuras/e1_mapa_calor.png` | `src/etapa1.py` |
| Conferência do Canal 2: 20 semanas sorteadas e nível no site (E1) | `results/tabelas/e1_canal2_semanas.csv`, `results/tabelas/e1_canal2_conferencia.csv` | `src/canal2.py` (leitura manual em `data/canal2_niveis_site.csv`) |
| Canal 3: API × boletim da Sesa 2024 (E1) | `results/tabelas/e1_canal3_comparacao.csv` | `src/canal3.py` (valores do boletim em `data/canal3_boletim_sesa_2024.csv`) |
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