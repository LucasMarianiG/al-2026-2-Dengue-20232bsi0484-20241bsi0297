# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/etapa3.py: P estimada, pi por 3 metodos (Serra, Vitoria, ES), iteracoes, primitividade/Wielandt, ocupacao empirica
"""Etapa 3: a distribuição estacionária, por três caminhos que precisam concordar.

Para cada território (Serra, Vitória e ES = soma dos 78 municípios):
  1. P̂ = estimador de máxima verossimilhança, p_ij = n_ij / n_i (sem suavização; a de Laplace é da E2);
  2. π por (i) sistema linear, (ii) potências explícitas a partir de e1, e4 e uniforme, com nº de
     iterações, e (iii) autovetor à esquerda;
  3. concordância entre métodos, resíduo ‖πP − π‖₁, menor k com P^k > 0 e limite de Wielandt;
  4. frequência empírica de ocupação (semanas por nível) para comparar com π.

Saídas: results/tabelas/e3_pi_metodos.csv, results/tabelas/e3_resumo.csv,
        results/figuras/e3_convergencia_serra.png.
Uso:  python -m src.etapa3      (ou via run_all.py)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dados import N_NIVEIS, PASTA_RAW, carregar_serie, contar_transicoes
from src.markov import (
    calcula_distribuicao_estacionaria_autovetor,
    calcula_distribuicao_estacionaria_potencias,
    calcula_distribuicao_estacionaria_sistema,
    limite_wielandt,
    menor_k_positivo,
    residuo_estacionario,
)

SERRA = 3205002
ESCOLHIDO = 3205309  # Vitória
TOL = 1e-12
METODOS = ["sistema", "autovetor", "potencias_e1", "potencias_e4", "potencias_uniforme", "freq_empirica"]


def matriz_transicao(C) -> np.ndarray:
    """P̂ = C / soma das linhas (máxima verossimilhança). Exige todas as linhas com pelo menos uma saída."""
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError(f"C deve ser quadrada; formato recebido: {C.shape}")
    if (C < 0).any():
        raise ValueError("contagens negativas")
    soma = C.sum(axis=1)
    if (soma == 0).any():
        raise ValueError(f"linha sem saídas observadas (estado(s) {np.flatnonzero(soma == 0) + 1}): P̂ indefinida")
    return C / soma[:, None]


def geocodes_do_estado(pasta=PASTA_RAW) -> list:
    """Os 78 geocódigos do ES, da lista congelada em data/raw/_municipios_ibge.csv."""
    return sorted(int(g) for g in pd.read_csv(Path(pasta) / "_municipios_ibge.csv")["geocode"])


def contagens_e_ocupacao(geocodes, pasta=PASTA_RAW):
    """Soma, sobre os municípios dados, da matriz de contagens C e das semanas por nível (ocupação)."""
    C = np.zeros((N_NIVEIS, N_NIVEIS), dtype=int)
    ocup = np.zeros(N_NIVEIS, dtype=int)
    for g in geocodes:
        serie = carregar_serie(g, pasta)
        C += contar_transicoes(serie)
        niveis = pd.to_numeric(serie["nivel"], errors="coerce").dropna().astype(int)
        ocup += np.bincount(niveis, minlength=N_NIVEIS + 1)[1:]
    return C, ocup


def trajetoria_erro(P, pi, pi0, k_max) -> np.ndarray:
    """‖π(k) − π‖₁ para k = 0..k_max, com π(k+1) = π(k)P a partir de pi0 (para a figura de convergência)."""
    atual = np.asarray(pi0, dtype=float)
    erros = [np.abs(atual - pi).sum()]
    for _ in range(k_max):
        atual = atual @ P
        erros.append(np.abs(atual - pi).sum())
    return np.array(erros)


def analisar(C, ocupacao) -> dict:
    """Roda os três métodos e as verificações para uma matriz de contagens. Devolve um dicionário plano."""
    P = matriz_transicao(C)
    n = P.shape[0]
    e1, e4, uni = np.eye(n)[0], np.eye(n)[-1], np.full(n, 1.0 / n)
    r = {"P": P}
    r["pi_sistema"] = calcula_distribuicao_estacionaria_sistema(P)
    r["pi_autovetor"] = calcula_distribuicao_estacionaria_autovetor(P)
    for nome, pi0 in (("e1", e1), ("e4", e4), ("uniforme", uni)):
        res = calcula_distribuicao_estacionaria_potencias(P, pi0=pi0, tol=TOL)
        r["pi_potencias_" + nome] = res.pi
        r["iteracoes_" + nome] = res.iteracoes
        r["convergiu_" + nome] = res.convergiu
    ocupacao = np.asarray(ocupacao, dtype=float)
    r["freq_empirica"] = ocupacao / ocupacao.sum()
    r["pi_freq_empirica"] = r["freq_empirica"]

    candidatos = [r["pi_" + m] for m in METODOS if m != "freq_empirica"]
    r["discordancia_max"] = max(np.abs(a - b).sum() for i, a in enumerate(candidatos) for b in candidatos[i + 1:])
    for m in METODOS:
        r["residuo_" + m] = residuo_estacionario(r["pi_" + m], P)
    r["l1_pi_vs_empirica"] = float(np.abs(r["pi_sistema"] - r["freq_empirica"]).sum())
    r["menor_k"] = menor_k_positivo(P)
    r["wielandt"] = limite_wielandt(n)
    modulos = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    r["lambda2"] = float(modulos[1])  # espectro completo e meia-vida ficam para a E4
    r["iteracoes_previstas"] = float(np.log(TOL) / np.log(r["lambda2"]))
    return r


def tabelas(resultados: dict, transicoes: dict) -> tuple:
    """Monta (tabela longa de π por método, tabela-resumo) a partir de {território: resultado}."""
    linhas, resumo = [], []
    for terr, r in resultados.items():
        for m in METODOS:
            ini = m.split("_", 1)[1] if m.startswith("potencias_") else None
            linhas.append({
                "territorio": terr, "metodo": m,
                **{f"pi_{i + 1}": r["pi_" + m][i] for i in range(N_NIVEIS)},
                "iteracoes": r.get("iteracoes_" + ini, "") if ini else "",
                "residuo": r["residuo_" + m],
            })
        resumo.append({
            "territorio": terr, "transicoes": transicoes[terr],
            "menor_k": r["menor_k"], "wielandt": r["wielandt"],
            "lambda2": r["lambda2"], "iteracoes_previstas": r["iteracoes_previstas"],
            "iteracoes_e1": r["iteracoes_e1"], "iteracoes_e4": r["iteracoes_e4"],
            "iteracoes_uniforme": r["iteracoes_uniforme"],
            "discordancia_max": r["discordancia_max"], "l1_pi_vs_empirica": r["l1_pi_vs_empirica"],
        })
    return pd.DataFrame(linhas), pd.DataFrame(resumo)


def figura_convergencia(r: dict, titulo: str, caminho, k_max=260) -> None:
    """‖π(k) − π‖₁ em escala log, partindo de e1, e4 e uniforme, com a referência |λ2|^k."""
    P, pi = r["P"], r["pi_sistema"]
    n = P.shape[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    estilos = {"e1": ("-", "black"), "e4": ("--", "0.35"), "uniforme": (":", "black")}
    for nome, pi0 in (("e1", np.eye(n)[0]), ("e4", np.eye(n)[-1]), ("uniforme", np.full(n, 1 / n))):
        ls, cor = estilos[nome]
        ax.semilogy(trajetoria_erro(P, pi, pi0, k_max), ls=ls, color=cor, label=f"início {nome} ({r['iteracoes_' + nome]} iterações)")
    ax.semilogy(2 * r["lambda2"] ** np.arange(k_max + 1), color="0.6", lw=3, alpha=0.6, label=f"2·|λ₂|^k, |λ₂| = {r['lambda2']:.4f}")
    ax.axhline(TOL, color="0.8", lw=0.8)
    ax.set_ylim(1e-14, 3)
    ax.set_xlabel("iteração k")
    ax.set_ylabel("‖π(k) − π‖₁")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    ax.grid(color="0.9", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def main(pasta_raw=PASTA_RAW, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    territorios = {"Serra": [SERRA], "Vitória": [ESCOLHIDO], "ES": geocodes_do_estado(pasta_raw)}
    resultados, transicoes = {}, {}
    for nome, geocodes in territorios.items():
        C, ocup = contagens_e_ocupacao(geocodes, pasta_raw)
        resultados[nome] = analisar(C, ocup)
        transicoes[nome] = int(C.sum())
        if resultados[nome]["discordancia_max"] >= 1e-10:
            raise SystemExit(f"Métodos discordam em {nome}: {resultados[nome]['discordancia_max']:.2e} >= 1e-10")

    longa, resumo = tabelas(resultados, transicoes)
    longa.to_csv(saida / "tabelas" / "e3_pi_metodos.csv", index=False, encoding="utf-8")
    resumo.to_csv(saida / "tabelas" / "e3_resumo.csv", index=False, encoding="utf-8")
    figura_convergencia(resultados["Serra"], "Serra: convergência do método das potências a π",
                        saida / "figuras" / "e3_convergencia_serra.png")

    for _, l in resumo.iterrows():
        print(f"E3 {l['territorio']}: {l['transicoes']} transicoes | menor k = {l['menor_k']} (Wielandt {l['wielandt']}) | "
              f"|l2| = {l['lambda2']:.4f} | iter e1/e4/unif = {l['iteracoes_e1']}/{l['iteracoes_e4']}/{l['iteracoes_uniforme']} | "
              f"discordancia = {l['discordancia_max']:.1e}")
    return longa


if __name__ == "__main__":
    main()
# [/IA]
