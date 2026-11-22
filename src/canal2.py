# [IA] Claude Sonnet 5.5 - 2026-11-22 - src/canal2.py: sorteio reprodutivel (semente 2026) de 10 semanas por municipio para a conferencia do Canal 2 (E1)
"""Canal 2 da Etapa 1: conferir o nível de alerta de 10 semanas sorteadas contra o site do InfoDengue.

Resposta do professor (22/11/2026): são 10 semanas PARA CADA município (Serra e o escolhido).
Protocolo (fixo, para o sorteio ser reproduzível):
  * um único gerador numpy.random.default_rng(2026);
  * municípios na ordem Serra, depois o escolhido;
  * candidatas = semanas da série congelada com nível registrado;
  * rng.choice(candidatas, 10, replace=False), apresentadas em ordem de data.

Saída: results/tabelas/e1_canal2_semanas.csv, com a coluna `nivel_no_site` VAZIA: é preenchida à mão ao
abrir a página `url` (a conferência no site não é automatizável nem reprodutível offline). As divergências
são explicadas no relatório (revisões retroativas, versão do modelo).
Uso:  python -m src.canal2
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.dados import PASTA_RAW, carregar_serie
from src.etapa1 import ESCOLHIDO, SERRA

SEMENTE = 2026
SEMANAS_POR_MUNICIPIO = 10
URL = "https://info.dengue.mat.br/alerta/{g}/dengue"


def sortear_semanas(series: dict, k=SEMANAS_POR_MUNICIPIO, semente=SEMENTE) -> pd.DataFrame:
    """Sorteia k semanas (sem reposição, entre as com nível registrado) por município, na ordem do dicionário."""
    rng = np.random.default_rng(semente)
    partes = []
    for geocode, serie in series.items():
        candidatas = serie[pd.to_numeric(serie["nivel"], errors="coerce").notna()].reset_index(drop=True)
        if len(candidatas) < k:
            raise ValueError(f"município {geocode}: só {len(candidatas)} semanas com nível; precisa de {k}")
        idx = np.sort(rng.choice(len(candidatas), size=k, replace=False))
        colunas = [c for c in ("data_iniSE", "SE", "nivel", "casos", "versao_modelo") if c in candidatas.columns]
        parte = candidatas.loc[idx, colunas].copy()
        parte.insert(0, "geocode", geocode)
        partes.append(parte)
    return pd.concat(partes, ignore_index=True)


def main(pasta_raw=PASTA_RAW, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    t = sortear_semanas({g: carregar_serie(g, pasta_raw) for g in (SERRA, ESCOLHIDO)})
    t["data_iniSE"] = pd.to_datetime(t["data_iniSE"]).dt.strftime("%Y-%m-%d")
    t["url"] = t["geocode"].map(lambda g: URL.format(g=g))
    t["nivel_no_site"] = np.nan
    t.to_csv(saida / "tabelas" / "e1_canal2_semanas.csv", index=False, encoding="utf-8")
    print(f"Canal 2: {len(t)} semanas sorteadas (semente {SEMENTE}); preencha nivel_no_site conferindo cada URL.")
    return t


if __name__ == "__main__":
    main()
# [/IA]
