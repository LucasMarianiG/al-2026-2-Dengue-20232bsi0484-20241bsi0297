# [IA] Claude Sonnet 5.5 - 2026-11-23 - src/etapa2.py: contagens por dois caminhos, P estimada, erros-padrao, suavizacao de Laplace, verificacoes e mapas de calor (E2)
"""Etapa 2: como estimar a matriz e quanto confiar nela?

Para cada território (Serra, Vitória e ES = soma dos 78 municípios):
  1. matriz de contagens C = (n_ij) por DOIS caminhos independentes (laço explícito e pandas.crosstab);
  2. P̂ por máxima verossimilhança, p̂_ij = n_ij / n_i, com erro-padrão aproximado sqrt(p̂_ij (1 − p̂_ij) / n_i);
  3. P̂ com suavização de Laplace, p̂_ij^(α) = (n_ij + α) / (n_i + kα), α = 0,5, k = 4 níveis, e o efeito da
     suavização sobre a distribuição estacionária, ‖π − π^(α)‖₁;
  4. verificações obrigatórias: P̂ ≥ 0, P̂1 = 1 (tolerância 1e-12) e soma(C) = transições válidas da Etapa 1.

Saídas: results/tabelas/e2_matrizes.csv, results/tabelas/e2_pi_laplace.csv, results/tabelas/e2_resumo.csv,
        results/figuras/e2_mapas_calor.png.
Uso:  python -m src.etapa2      (ou via run_all.py)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dados import N_NIVEIS, PASTA_RAW, carregar_serie, contar_transicoes
from src.etapa3 import ESCOLHIDO, SERRA, geocodes_do_estado, matriz_transicao
from src.markov import calcula_distribuicao_estacionaria_sistema, validar_estocastica

ALPHA = 0.5
TOL = 1e-12
AUDITORIA = Path("results/e1_auditoria.csv")


# --------------------------------------------------------------------------- contagens
def _extrair(serie, n_niveis):
    """Datas e níveis como arrays, validando a ordem das datas e o intervalo dos níveis (1..n_niveis)."""
    datas = pd.to_datetime(serie["data_iniSE"]).to_numpy()
    niveis = pd.to_numeric(serie["nivel"], errors="coerce").to_numpy(dtype=float)
    if len(datas) > 1 and not np.all(datas[1:] > datas[:-1]):
        raise ValueError("datas devem ser estritamente crescentes (ordene e remova duplicatas)")
    preenchidos = niveis[~np.isnan(niveis)]
    if np.any((preenchidos < 1) | (preenchidos > n_niveis) | (preenchidos % 1 != 0)):
        raise ValueError(f"nivel fora de 1..{n_niveis}")
    return datas, niveis


def contagens_por_laco(serie, n_niveis=N_NIVEIS) -> np.ndarray:
    """Caminho 1: C = (n_ij) por laço explícito sobre cada par de semanas vizinhas na tabela.

    O par (X_t, X_t+1) conta se as datas diferem em exatamente 7 dias e os dois níveis estão preenchidos;
    uma lacuna ou um nível ausente interrompe a sequência.
    """
    datas, niveis = _extrair(serie, n_niveis)
    sete_dias = np.timedelta64(7, "D")
    C = np.zeros((n_niveis, n_niveis), dtype=int)
    for t in range(len(datas) - 1):
        if datas[t + 1] - datas[t] != sete_dias:
            continue
        if np.isnan(niveis[t]) or np.isnan(niveis[t + 1]):
            continue
        C[int(niveis[t]) - 1, int(niveis[t + 1]) - 1] += 1
    return C


def contagens_por_crosstab(serie, n_niveis=N_NIVEIS) -> np.ndarray:
    """Caminho 2: os mesmos pares, mas escolhidos por máscara e tabulados com pandas.crosstab."""
    datas, niveis = _extrair(serie, n_niveis)
    if len(datas) < 2:
        return np.zeros((n_niveis, n_niveis), dtype=int)
    ok = (
        ((datas[1:] - datas[:-1]) == np.timedelta64(7, "D"))
        & ~np.isnan(niveis[:-1])
        & ~np.isnan(niveis[1:])
    )
    if not ok.any():
        return np.zeros((n_niveis, n_niveis), dtype=int)
    de = pd.Series(niveis[:-1][ok].astype(int), name="de")
    para = pd.Series(niveis[1:][ok].astype(int), name="para")
    tabela = pd.crosstab(de, para).reindex(
        index=range(1, n_niveis + 1), columns=range(1, n_niveis + 1), fill_value=0
    )
    return tabela.to_numpy(dtype=int)


# --------------------------------------------------------------------------- estimação e incerteza
def erros_padrao(C) -> np.ndarray:
    """Erro-padrão aproximado de cada p̂_ij: sqrt(p̂_ij (1 − p̂_ij) / n_i). Exige n_i > 0 em todas as linhas."""
    C = np.asarray(C, dtype=float)
    P = matriz_transicao(C)  # levanta ValueError se alguma linha não tiver saídas
    ni = C.sum(axis=1)
    return np.sqrt(P * (1.0 - P) / ni[:, None])


def suavizar_laplace(C, alpha=ALPHA) -> np.ndarray:
    """P̂^(α) = (n_ij + α) / (n_i + kα), k = número de níveis. Com α = 0 é o estimador de máxima verossimilhança."""
    if alpha < 0:
        raise ValueError(f"alpha deve ser não negativo; recebido {alpha}")
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C deve ser quadrada; formato recebido: {C.shape}")
    if (C < 0).any():
        raise ValueError("contagens negativas")
    k = C.shape[0]
    denominador = C.sum(axis=1) + k * alpha
    if (denominador == 0).any():
        raise ValueError("linha sem saídas observadas e alpha = 0: P̂ indefinida")
    return (C + alpha) / denominador[:, None]


def efeito_laplace(C, alpha=ALPHA) -> float:
    """‖π − π^(α)‖₁: quanto a suavização desloca a distribuição estacionária (π por sistema linear)."""
    pi = calcula_distribuicao_estacionaria_sistema(matriz_transicao(C))
    pi_alpha = calcula_distribuicao_estacionaria_sistema(suavizar_laplace(C, alpha))
    return float(np.abs(pi - pi_alpha).sum())


# --------------------------------------------------------------------------- verificação
def verificar(C, P, transicoes_validas, tol=TOL) -> None:
    """P ≥ 0, P1 = 1 (tolerância tol) e soma(C) = transições válidas da Etapa 1; levanta ValueError se falhar."""
    validar_estocastica(P, tol=tol)  # quadrada, sem negativos, linhas somando 1
    soma = int(np.asarray(C).sum())
    if soma != int(transicoes_validas):
        raise ValueError(f"soma de C ({soma}) diferente das transições válidas da Etapa 1 ({int(transicoes_validas)})")


# --------------------------------------------------------------------------- pipeline
def contagens_do_territorio(geocodes, pasta=PASTA_RAW):
    """Soma de C sobre os municípios dados, pelos dois caminhos; devolve (C, nº de transições pela Etapa 1).

    Se os dois caminhos discordarem em qualquer município, levanta ValueError.
    """
    C = np.zeros((N_NIVEIS, N_NIVEIS), dtype=int)
    ref = 0
    for g in geocodes:
        serie = carregar_serie(g, pasta)
        c1, c2 = contagens_por_laco(serie), contagens_por_crosstab(serie)
        if not np.array_equal(c1, c2):
            raise ValueError(f"caminhos de contagem discordam no município {g}")
        C += c1
        ref += int(contar_transicoes(serie).sum())
    return C, ref


def _transicoes_validas(geocodes, ref_calculada, auditoria):
    """Total de transições válidas da Etapa 1: da auditoria congelada, se existir; senão, recontado."""
    auditoria = Path(auditoria)
    if auditoria.exists():
        tab = pd.read_csv(auditoria).set_index("geocode")
        return int(tab.loc[list(geocodes), "transicoes_validas"].sum())
    return ref_calculada


def tabela_matrizes(territorio, C, P, ep, P_alpha) -> pd.DataFrame:
    """Tabela longa (de, para) com contagem, n_i, p̂, erro-padrão e p̂ suavizada."""
    n = C.shape[0]
    ni = C.sum(axis=1)
    return pd.DataFrame([
        {
            "territorio": territorio, "de": i + 1, "para": j + 1, "n_ij": int(C[i, j]), "n_i": int(ni[i]),
            "p_hat": P[i, j], "erro_padrao": ep[i, j], "p_laplace": P_alpha[i, j],
        }
        for i in range(n) for j in range(n)
    ])


def mapas_de_calor(resultados: dict, caminho) -> None:
    """Mapas de calor em escala de cinza: linha de cima P̂ (com erro-padrão), linha de baixo P̂ suavizada."""
    n_terr = len(resultados)
    fig, eixos = plt.subplots(2, n_terr, figsize=(4.2 * n_terr, 8), squeeze=False)

    def anotar(ax, M, E=None):
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                texto = f"{M[i, j]:.3f}".replace(".", ",")
                if E is not None:
                    texto += "\n±" + f"{E[i, j]:.3f}".replace(".", ",")
                ax.text(j, i, texto, ha="center", va="center", fontsize=8,
                        color="white" if M[i, j] > 0.55 else "black")

    for col, (nome, r) in enumerate(resultados.items()):
        for lin, (M, E, rotulo) in enumerate(((r["P"], r["ep"], "P̂ (± erro-padrão)"),
                                               (r["P_alpha"], None, f"P̂ com Laplace (α = {ALPHA})".replace(".", ",")))):
            ax = eixos[lin, col]
            im = ax.imshow(M, cmap="Greys", vmin=0, vmax=1)
            anotar(ax, M, E)
            ax.set_xticks(range(N_NIVEIS), labels=[str(k) for k in range(1, N_NIVEIS + 1)])
            ax.set_yticks(range(N_NIVEIS), labels=[str(k) for k in range(1, N_NIVEIS + 1)])
            ax.set_xlabel("nível na semana seguinte (j)")
            ax.set_ylabel("nível na semana atual (i)")
            ax.set_title(f"{nome}: {rotulo}", fontsize=10)
    fig.colorbar(im, ax=eixos, shrink=0.6, label="probabilidade de transição")
    fig.savefig(caminho, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(pasta_raw=PASTA_RAW, saida=Path("results"), auditoria=AUDITORIA) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    territorios = {"Serra": [SERRA], "Vitória": [ESCOLHIDO], "ES": geocodes_do_estado(pasta_raw)}
    resultados, resumo, matrizes, pis = {}, [], [], []
    for nome, geocodes in territorios.items():
        C, ref_recontada = contagens_do_territorio(geocodes, pasta_raw)
        transicoes = _transicoes_validas(geocodes, ref_recontada, auditoria)
        P = matriz_transicao(C)
        ep = erros_padrao(C)
        P_alpha = suavizar_laplace(C, ALPHA)
        verificar(C, P, transicoes)                           # P̂ ≥ 0, P̂1 = 1, soma(C) = transições válidas
        validar_estocastica(P_alpha, tol=TOL)                 # a suavizada também é estocástica
        pi = calcula_distribuicao_estacionaria_sistema(P)
        pi_alpha = calcula_distribuicao_estacionaria_sistema(P_alpha)
        delta = float(np.abs(pi - pi_alpha).sum())

        resultados[nome] = {"P": P, "ep": ep, "P_alpha": P_alpha}
        matrizes.append(tabela_matrizes(nome, C, P, ep, P_alpha))
        for i in range(N_NIVEIS):
            pis.append({"territorio": nome, "nivel": i + 1, "pi": pi[i], "pi_laplace": pi_alpha[i],
                        "diferenca": pi_alpha[i] - pi[i]})
        resumo.append({
            "territorio": nome, "soma_C": int(C.sum()), "transicoes_validas": transicoes,
            "transicoes_recontadas_e1": ref_recontada,
            "min_P": float(P.min()), "max_erro_soma_linhas": float(np.abs(P.sum(axis=1) - 1).max()),
            "zeros_estruturais": int((C == 0).sum()), "alpha": ALPHA, "delta_pi_laplace": delta,
        })

    pd.concat(matrizes, ignore_index=True).to_csv(saida / "tabelas" / "e2_matrizes.csv", index=False, encoding="utf-8")
    pd.DataFrame(pis).to_csv(saida / "tabelas" / "e2_pi_laplace.csv", index=False, encoding="utf-8")
    resumo = pd.DataFrame(resumo).rename(columns={"territorio": "territorio"})
    resumo.to_csv(saida / "tabelas" / "e2_resumo.csv", index=False, encoding="utf-8")
    mapas_de_calor(resultados, saida / "figuras" / "e2_mapas_calor.png")

    for _, l in resumo.iterrows():
        print(f"E2 {l['territorio']}: soma(C) = {l['soma_C']} (E1: {l['transicoes_validas']}) | "
              f"zeros em C = {l['zeros_estruturais']} | min P = {l['min_P']:.3f} | "
              f"erro nas linhas = {l['max_erro_soma_linhas']:.1e} | ‖π − π(α)‖₁ = {l['delta_pi_laplace']:.4f}")
    return resumo


if __name__ == "__main__":
    main()
# [/IA]