# [IA] Claude Sonnet 5.5 - 2026-11-23 - src/etapa6.py: cadeia absorvente (nivel 4), N, t, desvio-padrao, B, episodios 1->2 nos dados com IC binomial e serie de Neumann (E6)
"""Etapa 6: quando o amarelo acende, o vermelho virá?

Para cada território (Serra, Vitória e ES = soma dos 78 municípios), com P̂ por máxima verossimilhança:
  (a) nível 4 absorvente: N = (I − Q)⁻¹, t = N·1 e desvio-padrão do tempo até o primeiro vermelho, saindo de 1, 2 e 3;
  (b) transientes {2, 3} e absorventes {1, 4}: B = N R, a probabilidade de chegar ao vermelho antes de voltar ao verde;
  (c) conferência de (b) nos dados: episódios que começam na ENTRADA no nível 2 vindo do 1 e terminam na primeira
      visita a 1 ou a 4; proporção observada de episódios terminados em 4 × B[2→4], com IC binomial de 95%;
  (d) N pela inversa × soma parcial de Neumann Σ_{k=0}^{K} Q^k, para K crescente.

Verificações independentes (cada grandeza por mais de um caminho, discrepância registrada):
  * N: inversa (LU) × série de Neumann;
  * t: N·1 × solução de (I − Q)t = 1 × simulação com default_rng(2026);
  * B: N R × solução de (I − Q)B = R × simulação; B1 = 1;
  * episódios: laço explícito × expressão regular sobre a sequência de níveis;
  * IC de Wilson: fórmula escrita à mão × scipy.stats.binomtest(...).proportion_ci.

Critérios dos episódios (documentados no relatório):
  * Semanas consecutivas: data_iniSE difere em exatamente 7 dias; nível ausente ou lacuna interrompe o segmento.
  * Episódio: começa na semana t com X_t = 2 e X_{t−1} = 1 (mesmo segmento); termina na primeira semana t' > t com
    X_t' ∈ {1, 4}. Se o segmento acaba antes, o episódio é CENSURADO e excluído da proporção (é contado à parte).
  * Um episódio que termina em 1 e é seguido por 2 dá origem a um novo episódio (o 1 final é a "entrada" do próximo).
  * Entrar no 2 vindo do 1 não é o mesmo que "estar no 2": a diferença entre a proporção observada e B[2→4]
    é, em parte, um teste da ausência de memória (hipótese de Markov).

Saídas: results/tabelas/e6_vermelho.csv, e6_N.csv, e6_B.csv, e6_episodios.csv, e6_neumann.csv, e6_resumo.csv,
        results/figuras/e6_neumann.png, results/figuras/e6_B_empirico.png.
Uso:  python -m src.etapa6      (ou via run_all.py)
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.dados import N_NIVEIS, PASTA_RAW, carregar_serie
from src.etapa3 import ESCOLHIDO, SERRA, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao
from src.etapa5 import _segmentos
from src.markov import (
    matriz_fundamental_inversa,
    menor_m_norma_menor_que_1,
    neumann_ate_tolerancia,
    particao_absorvente,
    prob_absorcao,
    prob_absorcao_sistema,
    simular_absorcao,
    tempo_ate_absorcao,
    tempo_ate_absorcao_sistema,
    trajetoria_neumann,
    variancia_tempo_absorcao,
)

SEMENTE = 2026
N_SIMULACOES = 100_000
TOL_VERIFICACAO = 1e-10
TOL_NEUMANN = 1e-14
K_MAX_FIGURA = 3000
KS_TABELA = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
VERMELHO = [3]            # índice (0..3) do nível 4
VERDE_E_VERMELHO = [0, 3]  # absorventes de (b): níveis 1 e 4
NIVEL_INICIO = 2
Z95 = 1.959963984540054


# ----------------------------------------------------------------------------- (a) e (b): modelo
def analise_vermelho(P, rng) -> dict:
    """(a) Nível 4 absorvente: N (por dois caminhos), t (por três), variância e desvio-padrão."""
    transientes, _, Q, _ = particao_absorvente(P, VERMELHO)
    N = matriz_fundamental_inversa(Q)
    N_serie, K_serie = neumann_ate_tolerancia(Q, tol=TOL_NEUMANN)
    t = tempo_ate_absorcao(N)
    t_sistema = tempo_ate_absorcao_sistema(Q)
    var = variancia_tempo_absorcao(N, t)
    t_sim, dp_sim = [], []
    for i in transientes:
        tempos, _ = simular_absorcao(P, VERMELHO, i, N_SIMULACOES, rng)
        t_sim.append(tempos.mean())
        dp_sim.append(tempos.std(ddof=1))
    return {
        "transientes": transientes, "Q": Q, "N": N, "N_serie": N_serie, "K_serie": K_serie,
        "t": t, "t_sistema": t_sistema, "t_sim": np.array(t_sim), "var": var, "dp": np.sqrt(var),
        "dp_sim": np.array(dp_sim),
        "disc_N": float(np.abs(N - N_serie).max()), "disc_t": float(np.abs(t - t_sistema).max()),
        "m_Qm": menor_m_norma_menor_que_1(Q),
    }


def analise_B(P, rng) -> dict:
    """(b) Transientes {2, 3}, absorventes {1, 4}: B = N R por três caminhos; B1 = 1."""
    transientes, absorventes, Q, R = particao_absorvente(P, VERDE_E_VERMELHO)
    N = matriz_fundamental_inversa(Q)
    N_serie, K_serie = neumann_ate_tolerancia(Q, tol=TOL_NEUMANN)
    B = prob_absorcao(N, R)
    B_sistema = prob_absorcao_sistema(Q, R)
    B_sim = []
    for i in transientes:
        _, destinos = simular_absorcao(P, VERDE_E_VERMELHO, i, N_SIMULACOES, rng)
        B_sim.append((destinos == 3).mean())
    return {
        "transientes": transientes, "absorventes": absorventes, "Q": Q, "R": R, "N": N,
        "N_serie": N_serie, "K_serie": K_serie, "B": B, "B_sistema": B_sistema, "B_sim_vermelho": np.array(B_sim),
        "disc_N": float(np.abs(N - N_serie).max()), "disc_B": float(np.abs(B - B_sistema).max()),
        "erro_B1": float(np.abs(B.sum(axis=1) - 1.0).max()), "m_Qm": menor_m_norma_menor_que_1(Q),
    }


# ----------------------------------------------------------------------------- (c): episódios nos dados
def episodios_laco(L):
    """Caminho 1: (nº terminados em 1, nº terminados em 4, nº censurados) dos episódios 1→2 de um segmento."""
    fim1 = fim4 = cens = 0
    for t in range(1, len(L)):
        if L[t] == NIVEL_INICIO and L[t - 1] == 1:
            s = t + 1
            while s < len(L) and L[s] not in (1, 4):
                s += 1
            if s == len(L):
                cens += 1
            elif L[s] == 4:
                fim4 += 1
            else:
                fim1 += 1
    return fim1, fim4, cens


def episodios_regex(L):
    """Caminho 2: os mesmos episódios por expressão regular sobre a sequência de níveis como texto.

    A busca com "lookahead" permite que o 1 final de um episódio seja a entrada do seguinte.
    """
    texto = "".join(str(int(x)) for x in L)
    fim1 = fim4 = 0
    for m in re.finditer(r"(?=1(2[23]*)([14]))", texto):
        if m.group(2) == "4":
            fim4 += 1
        else:
            fim1 += 1
    cens = len(re.findall(r"(?=12[23]*$)", texto))
    return fim1, fim4, cens


def contar_episodios(segmentos) -> dict:
    """Soma dos episódios de uma lista de segmentos pelos dois caminhos; levanta ValueError se discordarem."""
    laco, regex = np.zeros(3, dtype=int), np.zeros(3, dtype=int)
    for L in segmentos:
        laco += episodios_laco(L)
        regex += episodios_regex(L)
    if not np.array_equal(laco, regex):
        raise ValueError(f"contagens de episódios discordam: laço {laco.tolist()} × regex {regex.tolist()}")
    return {"termina_1": int(laco[0]), "termina_4": int(laco[1]), "censurados": int(laco[2])}


def ic_wilson(k, n, z=Z95):
    """IC de Wilson para uma proporção k/n (escrito à mão). (nan, nan) se n = 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    meia = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return float(max(0.0, centro - meia)), float(min(1.0, centro + meia))


def ic_binomial(k, n) -> dict:
    """IC de 95% da proporção k/n: Wilson (à mão, conferido com scipy) e Clopper–Pearson exato (scipy)."""
    if n == 0:
        return {"wilson": (float("nan"),) * 2, "exato": (float("nan"),) * 2}
    w = ic_wilson(k, n)
    w_scipy = binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    if abs(w[0] - w_scipy.low) > 1e-9 or abs(w[1] - w_scipy.high) > 1e-9:
        raise ValueError(f"IC de Wilson difere do scipy: {w} × {(w_scipy.low, w_scipy.high)}")
    e = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
    return {"wilson": w, "exato": (float(e.low), float(e.high))}


# ----------------------------------------------------------------------------- pipeline
def segmentos_do_territorio(geocodes, pasta=PASTA_RAW) -> list:
    """Segmentos contíguos (arrays de níveis) de todos os municípios dados."""
    segmentos = []
    for g in geocodes:
        segmentos.extend(_segmentos(carregar_serie(g, pasta)))
    return segmentos


def analisar_territorio(nome, geocodes, rng, pasta=PASTA_RAW) -> dict:
    C, _ = contagens_e_ocupacao(geocodes, pasta)
    P = matriz_transicao(C)
    a, b = analise_vermelho(P, rng), analise_B(P, rng)
    epi = contar_episodios(segmentos_do_territorio(geocodes, pasta))
    n_ep = epi["termina_1"] + epi["termina_4"]
    ic = ic_binomial(epi["termina_4"], n_ep)
    prop = epi["termina_4"] / n_ep if n_ep else float("nan")
    idx2 = b["transientes"].index(NIVEL_INICIO - 1)
    B24 = float(b["B"][idx2, b["absorventes"].index(3)])
    return {"nome": nome, "C": C, "P": P, "a": a, "b": b, "epi": epi, "n_ep": n_ep, "prop": prop, "ic": ic, "B24": B24}


def verificar(r, tol=TOL_VERIFICACAO) -> None:
    """Verificações obrigatórias: B1 = 1 e concordância entre caminhos; levanta ValueError se falhar."""
    a, b = r["a"], r["b"]
    for rotulo, valor in (("B1 = 1", b["erro_B1"]), ("N (a) inversa × Neumann", a["disc_N"]),
                          ("N (b) inversa × Neumann", b["disc_N"]), ("t: N·1 × (I−Q)t=1", a["disc_t"]),
                          ("B: NR × (I−Q)B=R", b["disc_B"])):
        if valor > tol:
            raise ValueError(f"{r['nome']}: {rotulo} com discrepância {valor:.2e} > {tol:.0e}")


def tabela_vermelho(r) -> pd.DataFrame:
    a = r["a"]
    return pd.DataFrame([{
        "territorio": r["nome"], "nivel_inicial": i + 1, "t_inversa": a["t"][k], "t_sistema": a["t_sistema"][k],
        "t_simulado": a["t_sim"][k], "variancia": a["var"][k], "desvio_padrao": a["dp"][k],
        "desvio_padrao_simulado": a["dp_sim"][k], "disc_inversa_sistema": abs(a["t"][k] - a["t_sistema"][k]),
    } for k, i in enumerate(a["transientes"])])


def tabela_N(r) -> pd.DataFrame:
    linhas = []
    for caso, d in (("nivel4_absorvente", r["a"]), ("transientes_2_3", r["b"])):
        niv = [i + 1 for i in d["transientes"]]
        for x, de in enumerate(niv):
            for y, para in enumerate(niv):
                linhas.append({"territorio": r["nome"], "caso": caso, "de": de, "para": para,
                               "N_inversa": d["N"][x, y], "N_neumann": d["N_serie"][x, y],
                               "disc": abs(d["N"][x, y] - d["N_serie"][x, y])})
    return pd.DataFrame(linhas)


def tabela_B(r) -> pd.DataFrame:
    b = r["b"]
    return pd.DataFrame([{
        "territorio": r["nome"], "nivel_inicial": i + 1,
        "B_verde": b["B"][k, 0], "B_vermelho": b["B"][k, 1],
        "B_vermelho_sistema": b["B_sistema"][k, 1], "B_vermelho_simulado": b["B_sim_vermelho"][k],
        "soma_linha": b["B"][k].sum(),
    } for k, i in enumerate(b["transientes"])])


def linha_episodios(r) -> dict:
    ic, e = r["ic"], r["epi"]
    cobre = bool(ic["exato"][0] <= r["B24"] <= ic["exato"][1]) if r["n_ep"] else None
    return {
        "territorio": r["nome"], "episodios": r["n_ep"], "censurados": e["censurados"],
        "terminam_em_4": e["termina_4"], "terminam_em_1": e["termina_1"], "proporcao_observada": r["prop"],
        "B_2_para_4": r["B24"], "wilson_inf": ic["wilson"][0], "wilson_sup": ic["wilson"][1],
        "exato_inf": ic["exato"][0], "exato_sup": ic["exato"][1], "B_dentro_do_IC_exato": cobre,
    }


def tabela_neumann(r) -> pd.DataFrame:
    linhas = []
    for caso, d in (("nivel4_absorvente", r["a"]), ("transientes_2_3", r["b"])):
        erros = trajetoria_neumann(d["Q"], d["N"], max(KS_TABELA))
        linhas.extend({"territorio": r["nome"], "caso": caso, "K": K, "erro_inf": float(erros[K])} for K in KS_TABELA)
    return pd.DataFrame(linhas)


def figura_neumann(resultados, caminho) -> None:
    estilos = [("-", "o"), ("--", "s"), (":", "^")]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for (nome, r), (ls, mk) in zip(resultados.items(), estilos):
        for caso, d, cinza in (("nível 4 absorvente", r["a"], "0.0"), ("transientes {2, 3}", r["b"], "0.5")):
            erros = np.maximum(trajetoria_neumann(d["Q"], d["N"], K_MAX_FIGURA), 1e-17)
            ax.semilogy(np.arange(len(erros)), erros, linestyle=ls, color=cinza, marker=mk, markevery=300,
                        markersize=5, linewidth=1.2, label=f"{nome}, {caso}")
    ax.set_xlabel("K (número de termos da série)")
    ax.set_ylabel(r"$\|N - \sum_{k=0}^{K} Q^k\|_\infty$")
    ax.set_title("Convergência da série de Neumann para N = (I − Q)⁻¹")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def figura_B_empirico(resultados, caminho) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for x, (nome, r) in enumerate(resultados.items()):
        if r["n_ep"]:
            inf, sup = r["ic"]["exato"]
            ax.errorbar(x - 0.08, r["prop"], yerr=[[r["prop"] - inf], [sup - r["prop"]]], fmt="o", color="black",
                        capsize=4, label="proporção observada (IC exato 95%)" if x == 0 else None)
        ax.plot(x + 0.08, r["B24"], marker="s", color="0.45", linestyle="none", markersize=7,
                label="B[2→4] do modelo" if x == 0 else None)
        ax.annotate(f"n = {r['n_ep']}", (x, 0.02), ha="center", fontsize=8, textcoords="offset points", xytext=(0, 0))
    ax.set_xticks(range(len(resultados)), labels=list(resultados))
    ax.set_xlim(-0.6, len(resultados) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_ylabel("probabilidade de chegar ao nível 4 antes do nível 1")
    ax.set_title("Episódios iniciados na entrada em nível 2 vindo do 1")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(pasta_raw=PASTA_RAW, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMENTE)

    territorios = {"Serra": [SERRA], "Vitória": [ESCOLHIDO], "ES": geocodes_do_estado(pasta_raw)}
    resultados = {}
    for nome, geocodes in territorios.items():
        r = analisar_territorio(nome, geocodes, rng, pasta_raw)
        verificar(r)
        resultados[nome] = r

    tab = saida / "tabelas"
    pd.concat([tabela_vermelho(r) for r in resultados.values()], ignore_index=True).to_csv(tab / "e6_vermelho.csv", index=False, encoding="utf-8")
    pd.concat([tabela_N(r) for r in resultados.values()], ignore_index=True).to_csv(tab / "e6_N.csv", index=False, encoding="utf-8")
    pd.concat([tabela_B(r) for r in resultados.values()], ignore_index=True).to_csv(tab / "e6_B.csv", index=False, encoding="utf-8")
    episodios = pd.DataFrame([linha_episodios(r) for r in resultados.values()])
    episodios.to_csv(tab / "e6_episodios.csv", index=False, encoding="utf-8")
    pd.concat([tabela_neumann(r) for r in resultados.values()], ignore_index=True).to_csv(tab / "e6_neumann.csv", index=False, encoding="utf-8")
    resumo = pd.DataFrame([{
        "territorio": r["nome"], "disc_N_a": r["a"]["disc_N"], "disc_N_b": r["b"]["disc_N"],
        "disc_t": r["a"]["disc_t"], "disc_B": r["b"]["disc_B"], "erro_B1": r["b"]["erro_B1"],
        "termos_neumann_a": r["a"]["K_serie"], "termos_neumann_b": r["b"]["K_serie"],
        "menor_m_normaQm_a": r["a"]["m_Qm"], "menor_m_normaQm_b": r["b"]["m_Qm"],
        "simulacoes": N_SIMULACOES, "semente": SEMENTE,
    } for r in resultados.values()])
    resumo.to_csv(tab / "e6_resumo.csv", index=False, encoding="utf-8")
    figura_neumann(resultados, saida / "figuras" / "e6_neumann.png")
    figura_B_empirico(resultados, saida / "figuras" / "e6_B_empirico.png")

    for r in resultados.values():
        a, b = r["a"], r["b"]
        t1 = ", ".join(f"{x:.2f}" for x in a["t"])
        ic = r["ic"]["exato"]
        print(f"E6 {r['nome']}: t(1,2,3) = {t1} sem | B[2→4] = {r['B24']:.4f} | "
              f"observado = {r['prop']:.4f} ({r['epi']['termina_4']}/{r['n_ep']}, IC exato [{ic[0]:.3f}; {ic[1]:.3f}], "
              f"{r['epi']['censurados']} censurados) | disc N/t/B = {max(a['disc_N'], b['disc_N']):.1e}/"
              f"{a['disc_t']:.1e}/{b['disc_B']:.1e} | erro B1 = {b['erro_B1']:.1e}")
    return episodios


if __name__ == "__main__":
    main()
# [/IA]
