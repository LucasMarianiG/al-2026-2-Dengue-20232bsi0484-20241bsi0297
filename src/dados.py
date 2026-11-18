# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/dados.py: leitura da serie congelada e contagem de transicoes validas (E1)
"""Leitura dos dados congelados e contagem de transições válidas.

Definição (enunciado, Seção 5): transição válida é um par (X_t, X_t+1) em que as duas
semanas são consecutivas no calendário epidemiológico e ambas têm nível registrado.
Aqui "consecutivas" significa: data_iniSE difere em exatamente 7 dias. Isso resolve a
SE 53 sem tabela de calendário (202053 -> 202101 também são 7 dias).
"""
from pathlib import Path

import numpy as np
import pandas as pd

N_NIVEIS = 4
PASTA_RAW = Path("data/raw")


def ler_bruto(geocode, pasta=PASTA_RAW) -> pd.DataFrame:
    """Lê o CSV congelado do município, sem remover nada (preserva duplicatas).

    Devolve as colunas data_iniSE (datetime), SE, nivel e versao_modelo, em ordem de data.
    """
    caminho = Path(pasta) / f"{geocode}.csv"
    df = pd.read_csv(caminho, usecols=["data_iniSE", "SE", "nivel", "versao_modelo"])
    df["data_iniSE"] = pd.to_datetime(df["data_iniSE"])
    return df.sort_values(["data_iniSE", "versao_modelo"]).reset_index(drop=True)


def carregar_serie(geocode, pasta=PASTA_RAW) -> pd.DataFrame:
    """Série semanal do município: ordenada por data, uma linha por semana.

    Critério para duplicatas (mesma data_iniSE): fica a linha de versao_modelo mais
    recente, por ser a revisão mais atual do InfoDengue. Nos dados congelados não há
    duplicatas (conferido na auditoria); o critério existe para o caso de aparecerem.
    """
    df = ler_bruto(geocode, pasta)
    df = df.drop_duplicates(subset="data_iniSE", keep="last")  # df já está ordenado por versao_modelo
    return df.reset_index(drop=True)


def contar_transicoes(serie: pd.DataFrame, n_niveis: int = N_NIVEIS) -> np.ndarray:
    """Matriz de contagens C = (n_ij), de i para j, com i, j = 1..n_niveis (índice 0..n-1).

    Exige datas estritamente crescentes (sem duplicatas). Um par entra na contagem se:
    (a) as datas diferem em exatamente 7 dias e (b) os dois níveis estão preenchidos.
    Lacunas e níveis ausentes interrompem a sequência: nada atravessa uma semana ausente.
    """
    datas = pd.to_datetime(serie["data_iniSE"]).to_numpy()
    niveis = pd.to_numeric(serie["nivel"], errors="coerce").to_numpy(dtype=float)

    if len(datas) > 1 and not np.all(datas[1:] > datas[:-1]):
        raise ValueError("datas devem ser estritamente crescentes (ordene e remova duplicatas)")
    validos = ~np.isnan(niveis)
    if np.any((niveis[validos] < 1) | (niveis[validos] > n_niveis) | (niveis[validos] % 1 != 0)):
        raise ValueError(f"nivel fora de 1..{n_niveis}")

    C = np.zeros((n_niveis, n_niveis), dtype=int)
    if len(datas) < 2:
        return C
    consecutivas = (datas[1:] - datas[:-1]) == np.timedelta64(7, "D")
    ok = consecutivas & validos[:-1] & validos[1:]
    de = niveis[:-1][ok].astype(int) - 1
    para = niveis[1:][ok].astype(int) - 1
    np.add.at(C, (de, para), 1)
    return C
# [/IA]