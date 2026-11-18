# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/etapa4.py: espectro, meia-vida, P^52 por duas vias, kappa_1 e curvas de decaimento (Serra, Vitoria, ES)
"""Etapa 4: por quanto tempo o presente informa o futuro?

Para cada território (Serra, Vitória e ES = soma dos 78 municípios), com P̂ por máxima verossimilhança:
  1. os quatro autovalores (np.linalg.eig e, de forma independente, raízes do polinômio característico
     de Faddeev–LeVerrier), |λ2| e a meia-vida ln(1/2)/ln|λ2|;
  2. verificação: soma dos autovalores = traço e produto = determinante (tolerância 1e-10);
  3. diagonalização P = VΛV⁻¹, κ1(V) = ‖V‖1‖V⁻¹‖1 e P^52 por quadrados sucessivos (explícitos)
     e por VΛ^52V⁻¹, com a diferença em norma 1 (mais conferências por matrix_power e multiplicação repetida);
  4. curvas ‖e_i P^n − π‖₁ (i = 1..4) em escala logarítmica, com a reta de referência C|λ2|^n.

Saídas: results/tabelas/e4_espectro.csv, results/tabelas/e4_resumo.csv,
        results/figuras/e4_decaimento_{serra,vitoria,es}.png.
Uso:  python -m src.etapa4      (ou via run_all.py)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dados import PASTA_RAW
from src.etapa3 import ESCOLHIDO, SERRA, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao
from src.markov import (
    calcula_distribuicao_estacionaria_sistema,
    calcula_espectro,
    calcula_meia_vida,
    diagonaliza,
    espectro_por_polinomio,
    norma_1,
    numero_condicao_1,
    polinomio_caracteristico,
    potencia_por_diagonalizacao,
    potencia_por_multiplicacao,
    potencia_quadrados_sucessivos,
)

N_POTENCIA = 52
TOL_VERIFICACAO = 1e-10
# curvas distinguidas por tipo de linha e marcador (não por cor), legíveis em impressão preto e branco
ESTILOS = [
    {"linestyle": "-", "marker": "o", "color": "black"},
    {"linestyle": "--", "marker": "s", "color": "0.25"},
    {"linestyle": "-.", "marker": "^", "color": "black"},
    {"linestyle": ":", "marker": "D", "color": "0.4"},
]


def curvas_decaimento(P, pi, n_max) -> np.ndarray:
    """Matriz (4 × n_max+1) com ‖e_i P^n − π‖₁ para n = 0..n_max, por iteração explícita v ← vP."""
    P = np.asarray(P, dtype=float)
    k = P.shape[0]
    saida = np.empty((k, n_max + 1))
    for i in range(k):
        v = np.zeros(k)
        v[i] = 1.0
        saida[i, 0] = np.abs(v - pi).sum()
        for n in range(1, n_max + 1):
            v = v @ P
            saida[i, n] = np.abs(v - pi).sum()
    return saida


def constante_C(curvas, lambda2, piso=1e-10) -> float:
    """Menor C tal que C|λ2|^n majora todas as curvas (para pontos acima do piso numérico, longe do erro de arredondamento)."""
    n = np.arange(curvas.shape[1])
    razoes = curvas / lambda2 ** n
    validos = curvas > piso
    if not validos.any():
        raise ValueError("nenhum ponto acima do piso")
    return float(razoes[validos].max())


def analisar_matriz(P, pi=None) -> dict:
    """Espectro, memória, diagonalização, P^52 por vias independentes e curvas de decaimento de uma matriz P̂."""
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    if pi is None:
        pi = calcula_distribuicao_estacionaria_sistema(P)
    r = {"P": P, "pi": pi}

    # 1) espectro por dois caminhos independentes
    esp = calcula_espectro(P)
    esp_poli = espectro_por_polinomio(P)
    r["espectro"], r["espectro_polinomio"] = esp, esp_poli
    r["discrepancia_espectro"] = float(np.abs(esp - esp_poli).max())
    r["lambda2"] = float(abs(esp[1]))
    r["arg_lambda2"] = float(np.angle(complex(esp[1])))
    r["meia_vida"] = calcula_meia_vida(P)

    # 2) traço e determinante (autovalores x matriz; coeficientes de Faddeev–LeVerrier x numpy)
    traco, det = np.trace(P), np.linalg.det(P)
    c = polinomio_caracteristico(P)
    r["dif_traco"] = float(abs(esp.sum() - traco))
    r["dif_det"] = float(abs(np.prod(esp) - det))
    r["dif_traco_poli"] = float(abs(-c[1] - traco))
    r["dif_det_poli"] = float(abs((-1) ** n * c[n] - det))

    # 3) diagonalização e P^52 por três caminhos
    V, lam, Vinv = diagonaliza(P)
    r["kappa1"] = numero_condicao_1(V)
    r["residuo_diagonalizacao"] = norma_1(V @ np.diag(lam) @ Vinv - P)
    r["distancia_minima_autovalores"] = float(min(abs(lam[i] - lam[j]) for i in range(n) for j in range(i + 1, n)))
    q, produtos = potencia_quadrados_sucessivos(P, N_POTENCIA)
    d = potencia_por_diagonalizacao(V, lam, Vinv, N_POTENCIA)
    r["produtos_quadrados"] = produtos
    r["dif_P52_diagonalizacao"] = norma_1(q - d)
    r["imag_max_P52"] = float(np.abs(np.imag(d)).max())
    r["dif_P52_matrix_power"] = norma_1(q - np.linalg.matrix_power(P, N_POTENCIA))
    r["dif_P52_multiplicacao"] = norma_1(q - potencia_por_multiplicacao(P, N_POTENCIA))
    r["erro_P52_max"] = float(np.abs(q - pi).sum(axis=1).max())     # max_i ‖e_i P^52 − π‖₁

    # 4) curvas de decaimento e reta de referência
    if r["lambda2"] < 1:
        r["n_max"] = int(np.ceil(1.25 * np.log(1e-12) / np.log(r["lambda2"])))
    else:
        r["n_max"] = 500
    r["curvas"] = curvas_decaimento(P, pi, r["n_max"])
    r["C"] = constante_C(r["curvas"], r["lambda2"])
    return r


def figura_decaimento(r: dict, titulo: str, caminho) -> None:
    """‖e_i P^n − π‖₁ (i = 1..4) em escala log, com a reta C|λ2|^n; preto e branco (linha + marcador)."""
    curvas, lam2, C = r["curvas"], r["lambda2"], r["C"]
    n = np.arange(curvas.shape[1])
    passo = max(1, len(n) // 10)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for i, est in enumerate(ESTILOS):
        ax.semilogy(n, curvas[i], label=f"início $e_{i + 1}$", linewidth=1.1, markersize=4.5,
                    markevery=(i * passo // 4, passo), markerfacecolor="white", **est)
    ax.semilogy(n, C * lam2 ** n, color="0.65", linewidth=3.5, alpha=0.7, zorder=0,
                label=f"$C|\\lambda_2|^n$, $|\\lambda_2|$ = {lam2:.4f}, $C$ = {C:.2f}")
    ax.set_ylim(1e-14, 3)
    ax.set_xlabel("n (semanas)")
    ax.set_ylabel(r"$\|e_i\hat{P}^n - \pi\|_1$")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(color="0.9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def tabelas(resultados: dict) -> tuple:
    """(espectro em formato longo, resumo por território)."""
    espectro, resumo = [], []
    colunas = ["lambda2", "arg_lambda2", "meia_vida", "kappa1", "dif_P52_diagonalizacao", "imag_max_P52",
               "dif_P52_matrix_power", "dif_P52_multiplicacao", "produtos_quadrados", "erro_P52_max",
               "residuo_diagonalizacao", "distancia_minima_autovalores", "dif_traco", "dif_det",
               "dif_traco_poli", "dif_det_poli", "discrepancia_espectro", "C", "n_max"]
    for terr, r in resultados.items():
        for k, lam in enumerate(r["espectro"], start=1):
            lam = complex(lam)
            espectro.append({"territorio": terr, "k": k, "re": lam.real, "im": lam.imag, "modulo": abs(lam)})
        resumo.append({"territorio": terr, **{c: r[c] for c in colunas}})
    return pd.DataFrame(espectro), pd.DataFrame(resumo)


def main(pasta_raw=PASTA_RAW, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    territorios = {"Serra": [SERRA], "Vitória": [ESCOLHIDO], "ES": geocodes_do_estado(pasta_raw)}
    arquivos = {"Serra": "serra", "Vitória": "vitoria", "ES": "es"}
    resultados = {}
    for nome, geocodes in territorios.items():
        C, _ = contagens_e_ocupacao(geocodes, pasta_raw)
        r = analisar_matriz(matriz_transicao(C))
        resultados[nome] = r
        if max(r["dif_traco"], r["dif_traco_poli"]) >= TOL_VERIFICACAO:
            raise SystemExit(f"Soma dos autovalores diferente do traço em {nome}: {r['dif_traco']:.2e}")
        if max(r["dif_det"], r["dif_det_poli"]) >= TOL_VERIFICACAO:
            raise SystemExit(f"Produto dos autovalores diferente do determinante em {nome}: {r['dif_det']:.2e}")
        if r["discrepancia_espectro"] >= TOL_VERIFICACAO:
            raise SystemExit(f"Os dois caminhos do espectro discordam em {nome}: {r['discrepancia_espectro']:.2e}")
        if r["dif_P52_diagonalizacao"] >= TOL_VERIFICACAO:
            raise SystemExit(f"P^52 por quadrados e por diagonalização discordam em {nome}: {r['dif_P52_diagonalizacao']:.2e}")
        figura_decaimento(r, f"{nome}: decaimento de $\\|e_i\\hat{{P}}^n - \\pi\\|_1$",
                          saida / "figuras" / f"e4_decaimento_{arquivos[nome]}.png")

    espectro, resumo = tabelas(resultados)
    espectro.to_csv(saida / "tabelas" / "e4_espectro.csv", index=False, encoding="utf-8")
    resumo.to_csv(saida / "tabelas" / "e4_resumo.csv", index=False, encoding="utf-8")
    for _, l in resumo.iterrows():
        print(f"E4 {l['territorio']}: |l2| = {l['lambda2']:.4f} (arg {l['arg_lambda2']:.4f}) | "
              f"meia-vida = {l['meia_vida']:.2f} sem | kappa1(V) = {l['kappa1']:.2f} | "
              f"dif P^52 (quadrados x diag) = {l['dif_P52_diagonalizacao']:.1e} | "
              f"traco/det = {l['dif_traco']:.1e}/{l['dif_det']:.1e}")
    return resumo


if __name__ == "__main__":
    main()
# [/IA]
