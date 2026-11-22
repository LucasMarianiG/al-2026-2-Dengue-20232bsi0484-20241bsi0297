# [IA] Claude Sonnet 5.5 - 2026-11-22 - tests/test_canal3.py: testes da comparacao API x boletim de fechamento da Sesa 2024 (Canal 3, E1)
"""Testes de src/canal3.py. Valores do boletim lidos à mão do PDF da Sesa (SE 52/2024) e congelados em CSV."""
import csv
import glob

import numpy as np
import pandas as pd
import pytest

from src import canal3
from src.canal3 import ANO, comparar, incidencia_periodo, ler_boletim, somar_ano


def _api(n=52, pop=1000, casos=10, ano=2024):
    se = [ano * 100 + k for k in range(1, n + 1)]
    return pd.DataFrame({"SE": se, "casos": casos, "casprov": casos - 1, "casconf": 0, "pop": pop,
                         "p_inc100k": casos / pop * 1e5})


def test_ano_escolhido_e_2024():
    assert ANO == 2024


def test_boletim_congelado_tem_os_valores_do_pdf():
    b = ler_boletim()
    v = {(r.grandeza, r.periodo): r.valor for r in b.itertuples()}
    assert v[("casos_provaveis_ano", "SE01-SE52/2024")] == 241879
    assert v[("casos_confirmados_ano", "SE01-SE52/2024")] == 128943
    assert v[("incidencia_4semanas", "SE49-SE52/2024")] == 130.61
    assert (b["conferido_em"] == "2026-11-22").all()


def test_boletim_incidencia_4_semanas_e_a_soma_das_semanas():
    b = ler_boletim()
    sem = b[b["grandeza"] == "incidencia_semana"]["valor"].sum()
    assert sem == pytest.approx(130.61, abs=1e-9)  # o próprio PDF fecha: 29,39+27,66+37,84+35,72


def test_somar_ano_conta_so_o_ano_e_exige_52_semanas():
    d = pd.concat([_api(), _api(n=5, ano=2023)])
    assert somar_ano(d, 2024) == {"casos": 520, "casprov": 468, "casconf": 0}
    with pytest.raises(ValueError, match="semanas"):
        somar_ano(_api(n=51), 2024)


def test_incidencia_periodo_a_mao():
    d = _api(pop=1000, casos=10)
    # SE 49..52 = 4 semanas x 10 casos = 40 casos / 1000 hab x 1e5 = 4000
    assert incidencia_periodo(d, 202449, 202452) == pytest.approx(4000.0)
    with pytest.raises(ValueError, match="semanas"):
        incidencia_periodo(d, 202449, 202460)


def _csv_independente(g):
    """Segundo caminho: módulo csv, sem pandas."""
    soma = {"casos": 0.0, "casprov": 0.0}
    n = 0
    with open(f"data/raw/{g}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if 202401 <= int(r["SE"]) <= 202452:
                n += 1
                for c in soma:
                    soma[c] += float(r[c] or 0)
    return n, soma


def test_serra_2024_por_dois_caminhos():
    n, s = _csv_independente(3205002)
    api = pd.read_csv("data/raw/3205002.csv")
    assert n == 52 and somar_ano(api, 2024)["casos"] == s["casos"] == 21345
    assert somar_ano(api, 2024)["casprov"] == s["casprov"] == 20095


def test_es_soma_dos_78_municipios_por_dois_caminhos():
    arqs = sorted(glob.glob("data/raw/32*.csv"))
    assert len(arqs) == 78
    tot = {"casos": 0.0, "casprov": 0.0}
    for a in arqs:
        n, s = _csv_independente(a.split("/")[-1].split("\\")[-1][:-4])
        assert n == 52
        for c in tot:
            tot[c] += s[c]
    r = comparar()
    es = r[r["escopo"] == "ES"].set_index("grandeza")
    assert es.loc["casos_provaveis_ano", "api_casos"] == tot["casos"] == 256943
    assert es.loc["casos_provaveis_ano", "api_casprov"] == tot["casprov"] == 166407


def test_comparar_calcula_diferenca_relativa_e_marca_confirmados_indisponiveis():
    r = comparar()
    es = r[r["escopo"] == "ES"].set_index("grandeza")
    p = es.loc["casos_provaveis_ano"]
    assert p["boletim"] == 241879
    assert p["dif_rel_casos"] == pytest.approx((256943 - 241879) / 241879)
    assert p["dif_rel_casprov"] == pytest.approx((166407 - 241879) / 241879)
    c = es.loc["casos_confirmados_ano"]
    assert np.isnan(c["api_casos"]) and np.isnan(c["api_casprov"]) and "casconf" in c["observacao"]


def test_comparar_incidencia_de_serra_4_semanas():
    r = comparar()
    s = r[(r["grandeza"] == "incidencia_4semanas") & (r["escopo"] == "Serra")].iloc[0]
    assert s["boletim"] == 130.61
    assert s["api_casos"] == pytest.approx(734 / 546405 * 1e5)
    assert s["dif_rel_casos"] == pytest.approx((734 / 546405 * 1e5 - 130.61) / 130.61)


def test_comparar_traz_serra_anual_so_da_api_sem_boletim():
    r = comparar()
    a = r[(r["grandeza"] == "casos_ano") & (r["escopo"] == "Serra")].iloc[0]
    assert a["api_casos"] == 21345 and a["api_casprov"] == 20095
    assert np.isnan(a["boletim"]) and "não publica" in a["observacao"]


def test_main_grava_tabela(tmp_path):
    canal3.main(saida=tmp_path)
    t = pd.read_csv(tmp_path / "tabelas" / "e1_canal3_comparacao.csv")
    assert list(t.columns) == ["grandeza", "escopo", "periodo", "boletim", "api_casos", "api_casprov",
                               "dif_rel_casos", "dif_rel_casprov", "observacao"]
    assert len(t) == 2 + 4 + 1 + 1  # ES: 2 anuais; Serra: 4 semanas + 1 agregada (SE49-52) + 1 anual


def test_somar_ano_rejeita_semana_duplicada_mesmo_com_52_linhas():
    d = _api()
    d.loc[d.index[-1], "SE"] = d["SE"].iloc[0]  # 52 linhas, mas uma SE repetida e outra faltando
    with pytest.raises(ValueError, match="semanas"):
        somar_ano(d, 2024)


def test_incidencia_usa_a_populacao_da_ultima_semana_do_intervalo():
    d = _api(pop=1000, casos=10)
    d.loc[d["SE"] == 202452, "pop"] = 2000
    # 40 casos / 2000 hab x 1e5 = 2000 (e não 4000, que usaria a população da primeira semana)
    assert incidencia_periodo(d.sample(frac=1, random_state=1), 202449, 202452) == pytest.approx(2000.0)


def test_comparar_exige_78_municipios(tmp_path):
    import shutil
    for g in glob.glob("data/raw/32*.csv")[:5]:
        shutil.copy(g, tmp_path)
    with pytest.raises(ValueError, match="78"):
        comparar(tmp_path)


def test_incidencia_4_semanas_de_serra_por_casprov():
    r = comparar()
    s = r[(r["grandeza"] == "incidencia_4semanas") & (r["escopo"] == "Serra")].iloc[0]
    assert s["api_casprov"] == pytest.approx(730 / 546405 * 1e5)
    assert s["dif_rel_casprov"] == pytest.approx((730 / 546405 * 1e5 - 130.61) / 130.61)
# [/IA]

