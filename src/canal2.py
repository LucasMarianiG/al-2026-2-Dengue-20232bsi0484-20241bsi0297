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
MANUAL = Path("data/canal2_niveis_site.csv")  # leitura manual do site (cor), versionada
CORES = {"Verde": 1, "Amarelo": 2, "Laranja": 3, "Vermelho": 4}  # cor do site -> nível do InfoDengue; "?" = não conferida
URL = "https://info.dengue.mat.br/report/ES/{g}/{se}"  # relatório da semana AAAASS (a página /alerta/ só mostra a semana corrente)


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


def conferir(sorteio: pd.DataFrame, manual: pd.DataFrame) -> pd.DataFrame:
    """Compara o nível dos dados congelados com a cor lida à mão no site, semana a semana.

    `manual` tem geocode, SE, cor_site. Exige exatamente as semanas sorteadas (sem faltar, sobrar ou duplicar)
    e cores conhecidas (ou "?", que vira `sem_conferencia`). situacao: coincide | diverge | sem_conferencia.
    """
    chave = ["geocode", "SE"]
    if manual.duplicated(chave).any():
        raise ValueError("leitura manual com semanas duplicadas")
    desconhecidas = set(manual["cor_site"]) - set(CORES) - {"?"}
    if desconhecidas:
        raise ValueError(f"cor desconhecida na leitura manual: {sorted(desconhecidas)}")
    esperadas = set(zip(sorteio["geocode"], sorteio["SE"]))
    lidas = set(zip(manual["geocode"], manual["SE"]))
    if esperadas - lidas:
        raise ValueError(f"faltam na leitura manual: {sorted(esperadas - lidas)}")
    if lidas - esperadas:
        raise ValueError(f"sobram na leitura manual (não sorteadas): {sorted(lidas - esperadas)}")
    c = sorteio.merge(manual[chave + ["cor_site"]], on=chave, how="left")
    c["nivel_site"] = c["cor_site"].map(CORES).astype("Int64")
    c["nivel_dados"] = c["nivel"].astype("Int64")
    c["situacao"] = np.where(c["nivel_site"].isna(), "sem_conferencia",
                             np.where((c["nivel_site"] == c["nivel_dados"]).fillna(False).to_numpy(bool), "coincide", "diverge"))
    return c[["geocode", "SE", "data_iniSE", "nivel_dados", "nivel_site", "cor_site", "situacao", "versao_modelo", "url"]
            if "url" in c.columns else
            ["geocode", "SE", "data_iniSE", "nivel_dados", "nivel_site", "cor_site", "situacao", "versao_modelo"]]


def main(pasta_raw=PASTA_RAW, saida=Path("results"), manual=MANUAL) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    t = sortear_semanas({g: carregar_serie(g, pasta_raw) for g in (SERRA, ESCOLHIDO)})
    t["data_iniSE"] = pd.to_datetime(t["data_iniSE"]).dt.strftime("%Y-%m-%d")
    t["url"] = [URL.format(g=g, se=se) for g, se in zip(t["geocode"], t["SE"])]
    t["nivel_no_site"] = np.nan
    t.to_csv(saida / "tabelas" / "e1_canal2_semanas.csv", index=False, encoding="utf-8")
    print(f"Canal 2: {len(t)} semanas sorteadas (semente {SEMENTE}); preencha nivel_no_site conferindo cada URL.")
    if Path(manual).exists():
        c = conferir(t, pd.read_csv(manual))
        c.to_csv(saida / "tabelas" / "e1_canal2_conferencia.csv", index=False, encoding="utf-8")
        print(f"Canal 2: {(c['situacao'] == 'coincide').sum()} de {len(c)} coincidem com o site.")
    return t


if __name__ == "__main__":
    main()
# [/IA]
