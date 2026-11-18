# [IA] Claude Sonnet 5.5 - 2026-11-18 - testes da E1: carregar_serie e contar_transicoes (serie com lacuna feita a mao)
import numpy as np
import pandas as pd
import pytest

from src.dados import carregar_serie, contar_transicoes, ler_bruto

# Serie pequena construida a mao. Cruza a virada 2020/2021 (SE 202053 comeca em 27/12/2020)
# e tem uma LACUNA (semana de 10/01/2021 ausente).
#   data        SE      nivel
#   2020-12-13  202051  1
#   2020-12-20  202052  1
#   2020-12-27  202053  2     <- SE 53
#   2021-01-03  202101  2
#   (2021-01-10 ausente)      <- lacuna
#   2021-01-17  202103  3
#   2021-01-24  202104  4
#   2021-01-31  202105  4
DATAS = ["2020-12-13", "2020-12-20", "2020-12-27", "2021-01-03",
         "2021-01-17", "2021-01-24", "2021-01-31"]
NIVEIS = [1, 1, 2, 2, 3, 4, 4]


def serie_com_lacuna(niveis=NIVEIS):
    return pd.DataFrame({"data_iniSE": pd.to_datetime(DATAS), "nivel": niveis})


def test_lacuna_nao_e_atravessada():
    C = contar_transicoes(serie_com_lacuna())
    esperado = np.zeros((4, 4), dtype=int)
    esperado[0, 0] = 1  # 1->1
    esperado[0, 1] = 1  # 1->2 (ja dentro da SE 53)
    esperado[1, 1] = 1  # 2->2 (cruza a virada de ano, 202053 -> 202101)
    esperado[2, 3] = 1  # 3->4
    esperado[3, 3] = 1  # 4->4
    np.testing.assert_array_equal(C, esperado)
    assert C.sum() == 5          # 7 semanas -> 6 pares; 1 par cai na lacuna
    assert C[1, 2] == 0          # 2->3 atravessaria a lacuna: proibido


def test_virada_com_se53_e_contada():
    C = contar_transicoes(serie_com_lacuna())
    assert C[1, 1] == 1          # 202053 -> 202101 sao 7 dias: transicao valida


def test_nivel_ausente_interrompe_sequencia():
    s = serie_com_lacuna([1, 1, np.nan, 2, 3, 4, 4])
    C = contar_transicoes(s)
    assert C.sum() == 3          # so 1->1, 3->4, 4->4 restam
    assert C[0, 1] == 0 and C[1, 1] == 0


def test_serie_curta_ou_vazia():
    assert contar_transicoes(serie_com_lacuna().iloc[:1]).sum() == 0
    assert contar_transicoes(serie_com_lacuna().iloc[:0]).sum() == 0


def test_rejeita_datas_fora_de_ordem_ou_duplicadas():
    s = serie_com_lacuna()
    with pytest.raises(ValueError):
        contar_transicoes(s.iloc[::-1])
    with pytest.raises(ValueError):
        contar_transicoes(pd.concat([s, s.iloc[[0]]]).sort_values("data_iniSE"))


def test_rejeita_nivel_fora_de_1_a_4():
    with pytest.raises(ValueError):
        contar_transicoes(serie_com_lacuna([1, 1, 2, 2, 3, 4, 5]))


def _csv(tmp_path, linhas):
    cab = "data_iniSE,SE,nivel,versao_modelo\n"
    (tmp_path / "1234567.csv").write_text(cab + "\n".join(linhas) + "\n", encoding="utf-8")


def test_carregar_serie_ordena_e_remove_duplicata_pela_versao_mais_recente(tmp_path):
    _csv(tmp_path, [
        "2021-01-10,202102,2,2023-08-20",
        "2021-01-03,202101,1,2023-08-20",
        "2021-01-10,202102,3,2024-01-25",   # duplicata da mesma semana, versao mais recente
    ])
    s = carregar_serie(1234567, pasta=tmp_path)
    assert list(s["data_iniSE"].dt.strftime("%Y-%m-%d")) == ["2021-01-03", "2021-01-10"]
    assert list(s["nivel"]) == [1, 3]                       # ficou a versao mais recente
    assert len(ler_bruto(1234567, pasta=tmp_path)) == 3     # o bruto preserva a duplicata


def _contar_por_laco(datas, niveis):
    """Caminho 2 (ingenuo, sem pandas): conta transicoes com laco e datetime."""
    C = [[0] * 4 for _ in range(4)]
    for k in range(len(datas) - 1):
        if (datas[k + 1] - datas[k]).days == 7 and niveis[k] == niveis[k] and niveis[k + 1] == niveis[k + 1]:
            C[int(niveis[k]) - 1][int(niveis[k + 1]) - 1] += 1
    return np.array(C)


def test_serie_real_serra_dois_caminhos_concordam():
    s = carregar_serie(3205002)
    assert len(s) == 574
    C1 = contar_transicoes(s)
    C2 = _contar_por_laco(list(s["data_iniSE"].dt.to_pydatetime()), list(s["nivel"]))
    np.testing.assert_array_equal(C1, C2)
    assert C1.sum() == 573                                   # serie sem lacunas: 574 - 1
    assert int(s["nivel"].value_counts().sum()) == 574       # soma por nivel = total observado
# [/IA]
