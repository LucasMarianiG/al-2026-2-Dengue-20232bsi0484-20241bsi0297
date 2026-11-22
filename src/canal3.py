# [IA] Claude Sonnet 5.5 - 2026-11-22 - src/canal3.py: comparacao da API congelada com o boletim de fechamento da Sesa-ES 2024 (Canal 3, E1)
"""Canal 3 da Etapa 1: ordem de grandeza da API do InfoDengue contra o 52º boletim da dengue da Sesa-ES (2024).

O boletim publica, para o ES, os casos prováveis e confirmados do ano e, por município, só a incidência
(casos por 100 mil) das 4 últimas semanas. NÃO traz total anual de Serra. Por isso a conferência é:
  * ES anual: boletim x soma dos 78 municípios na API (colunas `casos` e `casprov`);
  * Serra SE 49-52: incidência do boletim x incidência recalculada da API (casos / pop x 1e5);
  * Serra anual: só o valor da API (declarado sem contraparte).
`casconf` vem zerada em toda a API: não é usada. Não se força coincidência; as diferenças são explicadas no relatório.
Uso:  python -m src.canal3
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from src.dados import PASTA_RAW
from src.etapa1 import SERRA

ANO = 2024
BOLETIM = Path("data/canal3_boletim_sesa_2024.csv")
SEMANAS_ANO = 52  # 2024 tem 52 semanas epidemiológicas (SE 1 a SE 52)
SE_FINAIS = (202449, 202452)  # as 4 últimas semanas do boletim
_COLUNAS = ["grandeza", "escopo", "periodo", "boletim", "api_casos", "api_casprov",
            "dif_rel_casos", "dif_rel_casprov", "observacao"]


def ler_boletim(caminho=BOLETIM) -> pd.DataFrame:
    return pd.read_csv(caminho, dtype={"conferido_em": str})


def _do_ano(df: pd.DataFrame, ano: int) -> pd.DataFrame:
    return df[(df["SE"] >= ano * 100 + 1) & (df["SE"] <= ano * 100 + 53)]


def somar_ano(df: pd.DataFrame, ano=ANO) -> dict:
    """Soma casos, casprov e casconf das semanas do ano; exige as 52 semanas (sem lacuna nem duplicata)."""
    d = _do_ano(df, ano)
    if len(d) != SEMANAS_ANO or d["SE"].duplicated().any():
        raise ValueError(f"{ano}: esperava {SEMANAS_ANO} semanas distintas, achei {len(d)}")
    return {c: d[c].sum() for c in ("casos", "casprov", "casconf")}


def incidencia_periodo(df: pd.DataFrame, se_ini: int, se_fim: int) -> float:
    """Casos por 100 mil hab. = soma de `casos` no intervalo / população x 1e5 (população da última semana)."""
    d = df[(df["SE"] >= se_ini) & (df["SE"] <= se_fim)].sort_values("SE")
    if len(d) != se_fim - se_ini + 1:
        raise ValueError(f"semanas {se_ini}-{se_fim}: esperava {se_fim - se_ini + 1}, achei {len(d)}")
    return float(d["casos"].sum() / d["pop"].iloc[-1] * 1e5)


def _rel(api, ref):
    return (api - ref) / ref if pd.notna(api) and pd.notna(ref) and ref != 0 else np.nan


def _linha(grandeza, escopo, periodo, boletim, a_casos, a_prov, obs=""):
    return {"grandeza": grandeza, "escopo": escopo, "periodo": periodo, "boletim": boletim,
            "api_casos": a_casos, "api_casprov": a_prov,
            "dif_rel_casos": _rel(a_casos, boletim), "dif_rel_casprov": _rel(a_prov, boletim), "observacao": obs}


def comparar(pasta_raw=PASTA_RAW, boletim=BOLETIM) -> pd.DataFrame:
    b = ler_boletim(boletim).set_index(["grandeza", "periodo"])["valor"]
    arqs = sorted(glob.glob(str(Path(pasta_raw) / "32*.csv")))
    if len(arqs) != 78:
        raise ValueError(f"esperava 78 municípios do ES, achei {len(arqs)}")
    es = {"casos": 0.0, "casprov": 0.0, "casconf": 0.0}
    for a in arqs:
        for c, v in somar_ano(pd.read_csv(a), ANO).items():
            es[c] += v
    per = f"SE01-SE52/{ANO}"
    linhas = [
        _linha("casos_provaveis_ano", "ES", per, b[("casos_provaveis_ano", per)], es["casos"], es["casprov"],
               "soma de 78 municípios; boletim = confirmados + em investigação"),
    ]
    conf_vazia = es["casconf"] == 0
    linhas.append(_linha("casos_confirmados_ano", "ES", per, b[("casos_confirmados_ano", per)],
                         np.nan if conf_vazia else es["casconf"], np.nan,
                         "casconf zerada em toda a API: indisponível" if conf_vazia else ""))
    serra = pd.read_csv(Path(pasta_raw) / f"{SERRA}.csv")
    for se in range(SE_FINAIS[0], SE_FINAIS[1] + 1):
        inc = incidencia_periodo(serra, se, se)
        linhas.append(_linha("incidencia_semana", "Serra", f"SE{se % 100}/{ANO}",
                             b[("incidencia_semana", f"SE{se % 100}/{ANO}")], inc, np.nan))
    p4 = f"SE49-SE52/{ANO}"
    linhas.append(_linha("incidencia_4semanas", "Serra", p4, b[("incidencia_4semanas", p4)],
                         incidencia_periodo(serra, *SE_FINAIS),
                         serra[(serra["SE"] >= SE_FINAIS[0]) & (serra["SE"] <= SE_FINAIS[1])]["casprov"].sum()
                         / serra["pop"].iloc[0] * 1e5))
    s = somar_ano(serra, ANO)
    linhas.append(_linha("casos_ano", "Serra", per, np.nan, s["casos"], s["casprov"],
                         "o boletim não publica total anual por município"))
    return pd.DataFrame(linhas, columns=_COLUNAS)


def main(pasta_raw=PASTA_RAW, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    t = comparar(pasta_raw)
    t.to_csv(saida / "tabelas" / "e1_canal3_comparacao.csv", index=False, encoding="utf-8")
    print(t.to_string(index=False))
    return t


if __name__ == "__main__":
    main()
# [/IA]
