# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_etapa1.py: testes da auditoria da E1 (calendario, auditoria, hashes, elegibilidade, graficos)
"""Testes da auditoria da Etapa 1 (src/etapa1.py).

Estrutura:
  1) calendário epidemiológico (independente dos dados);
  2) município pequeno construído à mão (lacuna, duplicata, nível ausente, SE 53);
  3) hashes do manifesto;
  4) elegibilidade;
  5) dados reais congelados, com um segundo caminho (csv + datetime, sem pandas).
"""
import csv
import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import etapa1

TEM_DADOS = Path("data/raw/3205002.csv").exists() and Path("data/MANIFEST.txt").exists()
precisa_dados = pytest.mark.skipif(not TEM_DADOS, reason="dados congelados ausentes")


# ---------------------------------------------------------------- 1) calendário
def test_semanas_esperadas_2015_2025():
    esp = etapa1.semanas_esperadas(2015, 2025)
    assert len(esp) == 574
    assert esp[0] == pd.Timestamp("2015-01-04")
    assert esp[-1] == pd.Timestamp("2025-12-28")
    assert (esp.dayofweek == 6).all()  # todas domingo
    assert (np.diff(esp.values) == np.timedelta64(7, "D")).all()


def test_anos_com_se_53_sao_2020_e_2025():
    esp = etapa1.semanas_esperadas(2015, 2025)
    por_ano = pd.Series(etapa1.codigo_se(esp) // 100).value_counts().sort_index()
    assert por_ano.loc[2020] == 53 and por_ano.loc[2025] == 53
    assert por_ano.drop([2020, 2025]).eq(52).all()


def test_codigo_se_viradas_de_ano():
    datas = pd.to_datetime(["2015-01-04", "2020-12-27", "2021-01-03", "2025-12-28"])
    assert etapa1.codigo_se(datas).tolist() == [201501, 202053, 202101, 202553]


# ---------------------------------------------------- 2) município pequeno à mão
# janela esperada: 7 domingos, de 2020-12-13 a 2021-01-24 (cruza a SE 53 de 2020)
ESPERADAS = pd.date_range("2020-12-13", "2021-01-24", freq="7D")
LINHAS = [  # fora de ordem de propósito
    ("2021-01-24", 202104, "", "2026-11-10"),     # nível ausente
    ("2020-12-13", 202051, 1, "2026-11-10"),
    ("2020-12-20", 202052, 4, "2026-10-01"),      # duplicata: versão antiga (descartada)
    ("2020-12-20", 202052, 2, "2026-11-10"),      # duplicata: versão recente (mantida)
    ("2020-12-27", 202053, 2, "2026-11-10"),      # SE 53
    ("2021-01-03", 202101, 3, "2026-11-10"),
    # 2021-01-10 ausente (lacuna)
    ("2021-01-17", 202103, 4, "2026-11-10"),
]


def _escreve(pasta, geocode, linhas):
    caminho = Path(pasta) / f"{geocode}.csv"
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["data_iniSE", "SE", "nivel", "versao_modelo"])
        w.writerows(linhas)


def test_auditoria_municipio_pequeno(tmp_path):
    _escreve(tmp_path, 9999999, LINHAS)
    a = etapa1.auditar_municipio(9999999, pasta=tmp_path, esperadas=ESPERADAS)
    assert a["semanas_esperadas"] == 7
    assert a["linhas_brutas"] == 7
    assert a["semanas_observadas"] == 6
    assert a["lacunas"] == 1
    assert a["duplicatas"] == 1
    assert a["datas_fora_do_calendario"] == 0
    assert a["se_inconsistentes"] == 0
    assert a["semanas_sem_nivel"] == 1
    # ficou a versão mais recente da duplicata (nível 2, não 4)
    assert [a[f"semanas_nivel_{k}"] for k in (1, 2, 3, 4)] == [1, 2, 1, 1]
    # pares válidos: 1->2, 2->2, 2->3 (cruza a SE 53); nada atravessa a lacuna nem o nível ausente
    assert a["transicoes_validas"] == 3
    assert a["transicoes_via_calendario"] == 3
    assert a["n_versoes_modelo"] == 2
    assert (a["versao_modelo_min"], a["versao_modelo_max"]) == ("2026-10-01", "2026-11-10")
    assert etapa1.verificar_linha(a) == []


def test_se_inconsistente_e_detectada(tmp_path):
    linhas = [list(l) for l in LINHAS]
    linhas[1][1] = 202052  # a data 2020-12-13 é SE 202051, não 202052
    _escreve(tmp_path, 9999999, linhas)
    a = etapa1.auditar_municipio(9999999, pasta=tmp_path, esperadas=ESPERADAS)
    assert a["se_inconsistentes"] == 1


def test_data_fora_do_calendario_e_detectada(tmp_path):
    linhas = LINHAS + [("2021-01-27", 202104, 1, "2026-11-10")]  # quarta-feira, fora da grade
    _escreve(tmp_path, 9999999, linhas)
    a = etapa1.auditar_municipio(9999999, pasta=tmp_path, esperadas=ESPERADAS)
    assert a["datas_fora_do_calendario"] == 1
    assert a["lacunas"] == 1  # a data extra não preenche a lacuna de 2021-01-10
    assert etapa1.verificar_linha(a) == []


def test_verificar_linha_acusa_inconsistencias(tmp_path):
    _escreve(tmp_path, 9999999, LINHAS)
    a = etapa1.auditar_municipio(9999999, pasta=tmp_path, esperadas=ESPERADAS)
    for campo, valor in [("semanas_nivel_2", 3), ("transicoes_validas", 4), ("lacunas", 0)]:
        ruim = dict(a, **{campo: valor})
        assert etapa1.verificar_linha(ruim) != [], campo


# ------------------------------------------------------------------- 3) hashes
def _monta_manifesto(pasta, nomes_e_conteudos):
    linhas = ["# cabeçalho", "# colunas"]
    for nome, conteudo in nomes_e_conteudos.items():
        (pasta / nome).write_bytes(conteudo)
        h = hashlib.sha256(conteudo).hexdigest()
        linhas.append(f"{nome}\t2026-11-15T00:00:00+00:00\thttp://x\t1\t{h}")
    manifesto = pasta.parent / "MANIFEST.txt"  # como no repositório: data/MANIFEST.txt, fora de data/raw/
    manifesto.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    return manifesto


def test_hashes_conferem_e_detectam_alteracao(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    man = _monta_manifesto(raw, {"a.csv": b"1,2\n", "b.csv": b"3,4\n"})
    assert etapa1.verificar_hashes(man, raw) == []
    (raw / "a.csv").write_bytes(b"1,2,X\n")  # alterado
    probs = etapa1.verificar_hashes(man, raw)
    assert len(probs) == 1 and "a.csv" in probs[0]


def test_hashes_detectam_arquivo_ausente_e_nao_listado(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    man = _monta_manifesto(raw, {"a.csv": b"1\n", "b.csv": b"2\n"})
    (raw / "b.csv").unlink()
    (raw / "c.csv").write_bytes(b"3\n")
    probs = " | ".join(etapa1.verificar_hashes(man, raw))
    assert "b.csv" in probs and "c.csv" in probs


# ------------------------------------------------------------- 4) elegibilidade
@pytest.mark.parametrize(
    "niveis, trans, esperado",
    [
        ((5, 5, 5, 5), 400, True),
        ((4, 5, 5, 5), 400, False),   # um nível com 4 semanas
        ((5, 5, 5, 5), 399, False),   # 399 transições
        ((300, 100, 0, 90), 500, False),
    ],
)
def test_eh_elegivel_nas_fronteiras(niveis, trans, esperado):
    linha = {f"semanas_nivel_{k}": v for k, v in zip((1, 2, 3, 4), niveis)}
    linha["transicoes_validas"] = trans
    assert etapa1.eh_elegivel(linha) is esperado


# ----------------------------------------------------------- gráficos (dados)
def test_fracao_nivel_alto_por_ano(tmp_path):
    linhas = [
        ("2020-12-27", 202053, 3, "2026-11-10"),  # ano 2020: 1 de 1 em nível >= 3
        ("2021-01-03", 202101, 1, "2026-11-10"),
        ("2021-01-10", 202102, 3, "2026-11-10"),
        ("2021-01-17", 202103, 4, "2026-11-10"),
        ("2021-01-24", 202104, 2, "2026-11-10"),
        ("2021-01-31", 202105, "", "2026-11-10"),  # sem nível: fora do denominador
    ]
    _escreve(tmp_path, 9999999, linhas)
    f = etapa1.fracao_nivel_alto_por_ano([9999999], pasta=tmp_path)
    assert f.loc[9999999, 2020] == pytest.approx(1.0)
    assert f.loc[9999999, 2021] == pytest.approx(0.5)


def test_figuras_sao_gravadas(tmp_path):
    _escreve(tmp_path, 9999999, LINHAS)
    serie = etapa1.carregar_serie(9999999, pasta=tmp_path)
    f1 = tmp_path / "serra.png"
    etapa1.figura_serie(serie, "Município de teste", f1)
    mat = pd.DataFrame({2020: [0.1, 0.9], 2021: [0.2, 0.8]}, index=[1, 2])
    f2 = tmp_path / "calor.png"
    etapa1.figura_mapa_calor(mat, {1: "A", 2: "B"}, f2)
    assert f1.stat().st_size > 1000 and f2.stat().st_size > 1000


# ----------------------------------------------- 5) dados reais + segundo caminho
def _auditar_com_csv_puro(geocode):
    """Segundo caminho: só csv e datetime, sem pandas e sem src.dados."""
    por_data = {}
    n_linhas = 0
    with open(Path("data/raw") / f"{geocode}.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n_linhas += 1
            d = date.fromisoformat(r["data_iniSE"])
            nivel = int(r["nivel"]) if r["nivel"].strip() != "" else None
            if d not in por_data or r["versao_modelo"] >= por_data[d][1]:
                por_data[d] = (nivel, r["versao_modelo"])
    datas = sorted(por_data)
    niveis = {k: sum(1 for d in datas if por_data[d][0] == k) for k in (1, 2, 3, 4)}
    trans = 0
    for d in datas:
        prox = d + timedelta(days=7)
        if prox in por_data and por_data[d][0] is not None and por_data[prox][0] is not None:
            trans += 1
    return {
        "linhas_brutas": n_linhas,
        "semanas_observadas": len(datas),
        "duplicatas": n_linhas - len(datas),
        **{f"semanas_nivel_{k}": v for k, v in niveis.items()},
        "transicoes_validas": trans,
    }


@precisa_dados
def test_serra_e_vitoria_reais():
    serra = etapa1.auditar_municipio(3205002)
    assert (serra["semanas_esperadas"], serra["semanas_observadas"]) == (574, 574)
    assert (serra["lacunas"], serra["duplicatas"], serra["semanas_sem_nivel"]) == (0, 0, 0)
    assert serra["transicoes_validas"] == 573
    assert [serra[f"semanas_nivel_{k}"] for k in (1, 2, 3, 4)] == [308, 129, 39, 98]
    vit = etapa1.auditar_municipio(3205309)
    assert vit["transicoes_validas"] == 573
    assert [vit[f"semanas_nivel_{k}"] for k in (1, 2, 3, 4)] == [294, 139, 48, 93]
    assert etapa1.eh_elegivel(serra) and etapa1.eh_elegivel(vit)


@precisa_dados
def test_hashes_dos_dados_congelados():
    assert etapa1.verificar_hashes() == []


@precisa_dados
def test_auditoria_dos_78_municipios_bate_com_o_segundo_caminho():
    df = etapa1.auditar_todos()
    assert len(df) == 78
    assert df["transicoes_validas"].sum() == 44694
    assert (df["se_inconsistentes"] == 0).all()
    assert (df["datas_fora_do_calendario"] == 0).all()
    for _, linha in df.iterrows():
        assert etapa1.verificar_linha(linha.to_dict()) == []
        outro = _auditar_com_csv_puro(int(linha["geocode"]))
        for campo, valor in outro.items():
            assert linha[campo] == valor, (linha["geocode"], campo)
    assert int(df["elegivel"].sum()) == 26


@precisa_dados
def test_main_grava_arquivos(tmp_path):
    etapa1.main(saida=tmp_path)
    aud = pd.read_csv(tmp_path / "e1_auditoria.csv")
    assert len(aud) == 78
    assert (tmp_path / "tabelas" / "e1_segundo_municipio.csv").exists()
    assert (tmp_path / "figuras" / "e1_serie_serra.png").stat().st_size > 1000
    assert (tmp_path / "figuras" / "e1_mapa_calor.png").stat().st_size > 1000
# [/IA]
