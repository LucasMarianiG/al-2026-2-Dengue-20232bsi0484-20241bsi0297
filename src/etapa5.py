# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/etapa5.py: permanencia, duracoes de corridas sem censura, histograma x geometrica, matriz M de primeira passagem por 2 caminhos e tempos empiricos 1->4 e 4->1
"""Etapa 5: quanto tempo se fica e quanto se demora a chegar?

Para cada território (Serra, Vitória e ES = soma dos 78 municípios), com P̂ por máxima verossimilhança:
  1. permanência média do modelo 1/(1 − p̂_ii) × duração observada das "corridas" (sequências máximas de
     semanas consecutivas no mesmo nível), EXCLUINDO as censuradas;
  2. nível 4: histograma das durações × distribuição geométrica implicada pelo modelo, P(D = d) = (1 − p̂44) p̂44^(d−1);
  3. matriz M = (m_ij) de primeira passagem por quatro sistemas 3×3 (um por estado-alvo j) e, de forma
     independente, pela matriz fundamental Z = (I − P + 1π)⁻¹ com m_ij = (z_jj − z_ij)/π_j; Kac m_jj = 1/π_j;
  4. tempos empíricos de primeira passagem 1→4 e 4→1, a comparar com m14 e m41.

Critérios (documentados no relatório):
  * Semanas consecutivas: data_iniSE difere em exatamente 7 dias. Um nível ausente ou uma lacuna interrompe a
    sequência: o dado vira uma sequência de "segmentos" contíguos, e nada atravessa a semana ausente.
  * Corrida: sequência máxima de semanas do mesmo nível dentro de um segmento. É CENSURADA se tocar o início
    ou o fim do segmento (não se sabe quando começou ou quando terminaria: a duração verdadeira é ≥ a observada).
  * Primeira passagem empírica i→j: para cada semana t em que a série está em i (modo "visitas") — ou apenas na
    primeira semana de cada corrida de i que não toca o início do segmento (modo "entradas") —, o tempo até a primeira
    semana t' > t em j, no mesmo segmento. Se o segmento acaba antes de chegar a j, a observação é censurada e excluída
    (isto subestima a média: só entram os tempos curtos o bastante para terminar dentro dos dados).

Saídas: results/tabelas/e5_permanencia.csv, e5_histograma_nivel4.csv, e5_M.csv, e5_resumo.csv,
        results/figuras/e5_hist_nivel4_{serra,vitoria,es}.png.
Uso:  python -m src.etapa5      (ou via run_all.py)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dados import N_NIVEIS, PASTA_RAW, carregar_serie
from src.etapa3 import ESCOLHIDO, SERRA, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao
from src.markov import (
    calcula_distribuicao_estacionaria_sistema,
    primeira_passagem_fundamental,
    primeira_passagem_sistemas,
    validar_estocastica,
)

TOL_VERIFICACAO = 1e-10
D_MAX = 12  # histograma do nível 4: durações 1..11 e a faixa "≥ 12"


# ----------------------------------------------------------------------------- segmentos e corridas
def _segmentos(serie) -> list:
    """Lista de arrays de níveis (inteiros) de segmentos contíguos: semanas a exatos 7 dias e com nível registrado."""
    datas = pd.to_datetime(serie["data_iniSE"]).to_numpy()
    niveis = pd.to_numeric(serie["nivel"], errors="coerce").to_numpy(dtype=float)
    segmentos, atual = [], []
    for k in range(len(niveis)):
        if np.isnan(niveis[k]):
            if atual:
                segmentos.append(np.array(atual, dtype=int))
            atual = []
            continue
        if atual and (datas[k] - datas[k - 1]) != np.timedelta64(7, "D"):
            segmentos.append(np.array(atual, dtype=int))
            atual = []
        atual.append(int(niveis[k]))
    if atual:
        segmentos.append(np.array(atual, dtype=int))
    return segmentos


def _corridas_do_segmento(L) -> list:
    """[(nivel, duracao, indice_inicial)] das corridas máximas de um segmento."""
    saida, inicio = [], 0
    for t in range(1, len(L) + 1):
        if t == len(L) or L[t] != L[inicio]:
            saida.append((int(L[inicio]), t - inicio, inicio))
            inicio = t
    return saida


def corridas(serie) -> pd.DataFrame:
    """Corridas da série: colunas nivel, duracao, censurada (True se toca o início ou o fim do segmento)."""
    linhas = []
    for L in _segmentos(serie):
        runs = _corridas_do_segmento(L)
        for idx, (nivel, dur, _) in enumerate(runs):
            linhas.append({"nivel": nivel, "duracao": dur, "censurada": idx == 0 or idx == len(runs) - 1})
    return pd.DataFrame(linhas, columns=["nivel", "duracao", "censurada"])


def duracoes_sem_censura(serie) -> dict:
    """{nivel: array de durações das corridas NÃO censuradas}, para os níveis 1..4."""
    c = corridas(serie)
    c = c[~c["censurada"]] if len(c) else c
    return {n: c.loc[c["nivel"] == n, "duracao"].to_numpy(dtype=int) for n in range(1, N_NIVEIS + 1)}


# ----------------------------------------------------------------------------- primeira passagem empírica
def tempos_primeira_passagem(serie, i, j, desde="visitas"):
    """Tempos empíricos (em semanas) de primeira passagem de i para j. Devolve (array de tempos, nº de censurados)."""
    if i == j:
        raise ValueError("i e j devem ser diferentes (o retorno é tratado por Kac)")
    if desde not in ("visitas", "entradas"):
        raise ValueError("desde deve ser 'visitas' ou 'entradas'")
    tempos, censurados = [], 0
    for L in _segmentos(serie):
        proximo_j = None  # índice da primeira ocorrência de j estritamente depois de t
        for t in range(len(L) - 1, -1, -1):
            conta = L[t] == i and (desde == "visitas" or (t > 0 and L[t - 1] != i))
            if conta:
                if proximo_j is None:
                    censurados += 1
                else:
                    tempos.append(proximo_j - t)
            if L[t] == j:
                proximo_j = t
    return np.array(tempos, dtype=int), censurados


# ----------------------------------------------------------------------------- modelo
def permanencia_media(P) -> np.ndarray:
    """Permanência média 1/(1 − p_ii) em cada nível (média da geométrica de parâmetro 1 − p_ii)."""
    P = validar_estocastica(P)
    pii = np.diag(P)
    if (pii >= 1.0).any():
        raise ValueError("estado absorvente (p_ii = 1): permanência infinita")
    return 1.0 / (1.0 - pii)


def histograma_geometrica(duracoes, p, d_max=D_MAX) -> pd.DataFrame:
    """Durações observadas × geométrica P(D = d) = (1 − p) p^(d−1). A última faixa é "≥ d_max" (cauda p^(d_max−1))."""
    duracoes = np.asarray(duracoes, dtype=int)
    if duracoes.size == 0:
        raise ValueError("sem durações para o histograma")
    N = duracoes.size
    d = np.arange(1, d_max + 1)
    observado = np.array([(duracoes == k).sum() for k in d[:-1]] + [(duracoes >= d_max).sum()])
    prob = np.array([(1 - p) * p ** (k - 1) for k in d[:-1]] + [p ** (d_max - 1)])
    return pd.DataFrame({"d": d, "observado": observado, "esperado": N * prob,
                         "frac_observada": observado / N, "frac_geometrica": prob})


def analisar_matriz(P, pi=None) -> dict:
    """M por dois caminhos independentes, Kac e as verificações obrigatórias para uma P̂."""
    P = validar_estocastica(P)
    if pi is None:
        pi = calcula_distribuicao_estacionaria_sistema(P)
    r = {"P": P, "pi": pi}
    r["permanencia"] = permanencia_media(P)
    r["M_sistemas"] = primeira_passagem_sistemas(P)
    r["M_fundamental"] = primeira_passagem_fundamental(P, pi)
    # Kac só é verificação de verdade na matriz resolvida pelos sistemas (na fórmula de Z a diagonal é 1/π por construção)
    r["kac_max"] = float(np.abs(np.diag(r["M_sistemas"]) - 1.0 / pi).max())
    r["dif_sistemas_fundamental"] = float(np.abs(r["M_sistemas"] - r["M_fundamental"]).max())
    r["m14"] = float(r["M_sistemas"][0, 3])
    r["m41"] = float(r["M_sistemas"][3, 0])
    return r


def dados_empiricos(geocodes, pasta=PASTA_RAW) -> dict:
    """Durações sem censura, nº de corridas censuradas e tempos empíricos 1→4 e 4→1, somando os municípios dados."""
    dur = {n: [] for n in range(1, N_NIVEIS + 1)}
    cens = {n: 0 for n in range(1, N_NIVEIS + 1)}
    fp = {(i, j, modo): ([], 0) for (i, j) in ((1, 4), (4, 1)) for modo in ("visitas", "entradas")}
    for g in geocodes:
        serie = carregar_serie(g, pasta)
        c = corridas(serie)
        for n in range(1, N_NIVEIS + 1):
            dur[n].extend(c.loc[(c["nivel"] == n) & ~c["censurada"], "duracao"].tolist())
            cens[n] += int(((c["nivel"] == n) & c["censurada"]).sum())
        for chave in fp:
            t, k = tempos_primeira_passagem(serie, *chave[:2], desde=chave[2])
            fp[chave][0].extend(t.tolist())
            fp[chave] = (fp[chave][0], fp[chave][1] + k)
    return {"duracoes": {n: np.array(v, dtype=int) for n, v in dur.items()}, "censuradas": cens, "passagem": fp}


# ----------------------------------------------------------------------------- saídas
def tabelas(resultados: dict, empiricos: dict) -> tuple:
    perm, hist, Mlinhas, resumo = [], [], [], []
    for terr, r in resultados.items():
        e = empiricos[terr]
        for n in range(1, N_NIVEIS + 1):
            d = e["duracoes"][n]
            perm.append({
                "territorio": terr, "nivel": n, "p_ii": r["P"][n - 1, n - 1],
                "permanencia_modelo": r["permanencia"][n - 1], "n_corridas_validas": len(d),
                "media_observada": float(d.mean()) if len(d) else np.nan,
                "mediana_observada": float(np.median(d)) if len(d) else np.nan,
                "maximo_observado": int(d.max()) if len(d) else np.nan,
                "n_corridas_censuradas": e["censuradas"][n],
            })
        h = histograma_geometrica(e["duracoes"][4], r["P"][3, 3])
        h.insert(0, "territorio", terr)
        hist.append(h)
        for i in range(N_NIVEIS):
            for j in range(N_NIVEIS):
                Mlinhas.append({"territorio": terr, "i": i + 1, "j": j + 1,
                                "m_sistemas": r["M_sistemas"][i, j], "m_fundamental": r["M_fundamental"][i, j],
                                "diferenca": abs(r["M_sistemas"][i, j] - r["M_fundamental"][i, j])})
        linha = {"territorio": terr, "kac_max": r["kac_max"], "dif_sistemas_fundamental": r["dif_sistemas_fundamental"],
                 "m14": r["m14"], "m41": r["m41"]}
        for (i, j), rot in (((1, 4), "14"), ((4, 1), "41")):
            for modo in ("visitas", "entradas"):
                t, k = e["passagem"][(i, j, modo)]
                t = np.asarray(t)
                linha[f"emp{rot}_{modo}_media"] = float(t.mean()) if len(t) else np.nan
                linha[f"emp{rot}_{modo}_n"] = len(t)
                linha[f"emp{rot}_{modo}_censurados"] = k
        resumo.append(linha)
    return pd.DataFrame(perm), pd.concat(hist, ignore_index=True), pd.DataFrame(Mlinhas), pd.DataFrame(resumo)


def figura_histograma(h: pd.DataFrame, titulo: str, caminho) -> None:
    """Barras cinza (observado) × pontos e linha pretos (geométrica), em frações; só preto-e-branco."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(h["d"], h["frac_observada"], width=0.7, color="0.75", edgecolor="black", hatch="//", label="observado")
    ax.plot(h["d"], h["frac_geometrica"], color="black", linestyle="--", marker="o", label="geométrica do modelo")
    ax.set_xticks(h["d"])
    ax.set_xticklabels([str(int(d)) for d in h["d"][:-1]] + [f"≥{int(h['d'].iloc[-1])}"])
    ax.set_xlabel("duração da corrida no nível 4 (semanas)")
    ax.set_ylabel("proporção das corridas")
    ax.set_title(titulo)
    ax.legend()
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def main(pasta_raw=PASTA_RAW, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    territorios = {"Serra": [SERRA], "Vitória": [ESCOLHIDO], "ES": geocodes_do_estado(pasta_raw)}
    arquivos = {"Serra": "serra", "Vitória": "vitoria", "ES": "es"}
    resultados, empiricos = {}, {}
    for nome, geocodes in territorios.items():
        C, _ = contagens_e_ocupacao(geocodes, pasta_raw)
        r = analisar_matriz(matriz_transicao(C))
        if r["kac_max"] >= TOL_VERIFICACAO:
            raise SystemExit(f"Fórmula de Kac violada em {nome}: {r['kac_max']:.2e}")
        if r["dif_sistemas_fundamental"] >= TOL_VERIFICACAO:
            raise SystemExit(f"M por sistemas e por Z discordam em {nome}: {r['dif_sistemas_fundamental']:.2e}")
        resultados[nome] = r
        empiricos[nome] = dados_empiricos(geocodes, pasta_raw)

    perm, hist, M, resumo = tabelas(resultados, empiricos)
    perm.to_csv(saida / "tabelas" / "e5_permanencia.csv", index=False, encoding="utf-8")
    hist.to_csv(saida / "tabelas" / "e5_histograma_nivel4.csv", index=False, encoding="utf-8")
    M.to_csv(saida / "tabelas" / "e5_M.csv", index=False, encoding="utf-8")
    resumo.to_csv(saida / "tabelas" / "e5_resumo.csv", index=False, encoding="utf-8")
    for nome in territorios:
        figura_histograma(hist[hist["territorio"] == nome].reset_index(drop=True),
                          f"{nome}: durações no nível 4 × geométrica",
                          saida / "figuras" / f"e5_hist_nivel4_{arquivos[nome]}.png")
    for _, l in resumo.iterrows():
        print(f"E5 {l['territorio']}: Kac {l['kac_max']:.1e} | M sist x Z {l['dif_sistemas_fundamental']:.1e} | "
              f"m14 = {l['m14']:.2f} (emp. {l['emp14_visitas_media']:.2f}, n={l['emp14_visitas_n']}) | "
              f"m41 = {l['m41']:.2f} (emp. {l['emp41_visitas_media']:.2f}, n={l['emp41_visitas_n']})")
    return resumo


if __name__ == "__main__":
    main()
# [/IA]
