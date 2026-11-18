# [IA] Claude Sonnet 5.5 - 2026-11-18 - tests/test_etapa5.py: testes do pipeline da Etapa 5 (corridas sem censura, primeira passagem empirica, histograma x geometrica, M por 2 caminhos)
"""Testes de src/etapa5.py: permanência, durações observadas e primeira passagem empírica."""
import itertools

import numpy as np
import pandas as pd
import pytest

from src import etapa5
from src.dados import carregar_serie
from src.etapa3 import SERRA, ESCOLHIDO, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao
from src.etapa5 import (
    corridas,
    duracoes_sem_censura,
    histograma_geometrica,
    tempos_primeira_passagem,
)


def _serie(niveis, lacunas=()):
    """Série semanal sintética: semanas consecutivas, saltando uma semana depois dos índices em `lacunas`."""
    datas, d = [], pd.Timestamp("2020-01-05")
    for k in range(len(niveis)):
        datas.append(d)
        d += pd.Timedelta(days=14 if k in lacunas else 7)
    return pd.DataFrame({"data_iniSE": datas, "nivel": niveis})


# ------------------------------------------------------------------ corridas e censura
def test_corridas_a_mao_marca_censura_de_inicio_e_de_fim():
    c = corridas(_serie([1, 1, 2, 2, 2, 3, 3, 1]))
    assert list(c["nivel"]) == [1, 2, 3, 1]
    assert list(c["duracao"]) == [2, 3, 2, 1]
    assert list(c["censurada"]) == [True, False, False, True]


def test_lacuna_interrompe_a_corrida_e_censura_os_dois_lados():
    # 2,2 | lacuna | 2,2: duas corridas de nível 2, ambas censuradas (nada atravessa a semana ausente)
    c = corridas(_serie([2, 2, 2, 2], lacunas=(1,)))
    assert list(c["duracao"]) == [2, 2]
    assert list(c["censurada"]) == [True, True]


def test_lacuna_no_meio_deixa_corrida_interna_sem_censura():
    # 1,2,2,3 | lacuna | 3,1,1,2,4: a corrida de 2 (índices 1-2) é interna ao 1º segmento
    c = corridas(_serie([1, 2, 2, 3, 3, 1, 1, 2, 4], lacunas=(3,)))
    interna = c[(c["nivel"] == 2) & (c["duracao"] == 2)]
    assert len(interna) == 1 and not interna.iloc[0]["censurada"]
    # internas: 2 (dur. 2) no 1º segmento; 1 (dur. 2) e 2 (dur. 1) no 2º segmento
    assert list(c["censurada"]).count(False) == 3


def test_nivel_ausente_interrompe_como_lacuna():
    c = corridas(_serie([1, 2, 2, np.nan, 2, 2, 3]))
    # segmentos [1,2,2] e [2,2,3]: cada corrida toca o começo ou o fim de um segmento
    assert list(c["duracao"]) == [1, 2, 2, 1]
    assert list(c["censurada"]) == [True, True, True, True]


def test_duracoes_somam_o_numero_de_semanas_com_nivel():
    s = _serie([1, 1, 2, 3, 3, 3, 4, np.nan, 4, 1], lacunas=(4,))
    assert corridas(s)["duracao"].sum() == s["nivel"].notna().sum()


def test_serie_curta_e_vazia():
    assert len(corridas(_serie([]))) == 0
    c = corridas(_serie([3]))
    assert list(c["censurada"]) == [True]


def test_duracoes_sem_censura_agrupa_por_nivel_e_exclui_censuradas():
    d = duracoes_sem_censura(_serie([1, 1, 2, 2, 2, 3, 3, 1, 4, 4, 1]))
    assert set(d) == {1, 2, 3, 4}
    assert list(d[2]) == [3] and list(d[3]) == [2] and list(d[4]) == [2]
    # nível 1: a corrida do início e a do fim são censuradas; só a do meio (vizinhos 3 e 4) conta
    assert list(d[1]) == [1]


# ------------------------------------------------------------------ primeira passagem empírica
def test_primeira_passagem_a_mao_desde_visitas():
    s = _serie([1, 1, 2, 3, 4, 1, 2, 4])
    t, cens = tempos_primeira_passagem(s, 1, 4)
    assert sorted(t) == [2, 3, 4] and cens == 0
    assert np.mean(t) == pytest.approx(3.0)
    t41, cens41 = tempos_primeira_passagem(s, 4, 1)
    assert list(t41) == [1] and cens41 == 1  # a visita ao 4 no fim não vê o 1: censurada


def test_primeira_passagem_desde_entradas_exclui_corrida_censurada_no_inicio():
    s = _serie([1, 1, 2, 3, 4, 1, 2, 4])
    t, cens = tempos_primeira_passagem(s, 1, 4, desde="entradas")
    assert list(t) == [2] and cens == 0  # só a corrida de 1 que começa após o 4 (índice 5)


def test_primeira_passagem_nao_atravessa_lacuna():
    s = _serie([1, 2, 3, 4], lacunas=(1,))  # lacuna entre os índices 1 e 2
    t, cens = tempos_primeira_passagem(s, 1, 4)
    assert len(t) == 0 and cens == 1


def test_primeira_passagem_ignora_a_propria_semana_ao_contar_o_alvo():
    # visita ao nível 1 seguida imediatamente do 4: tempo 1, não 0
    t, _ = tempos_primeira_passagem(_serie([1, 4]), 1, 4)
    assert list(t) == [1]


def test_primeira_passagem_rejeita_i_igual_a_j_e_modo_invalido():
    s = _serie([1, 2, 1])
    with pytest.raises(ValueError):
        tempos_primeira_passagem(s, 1, 1)
    with pytest.raises(ValueError):
        tempos_primeira_passagem(s, 1, 2, desde="qualquer")


# ------------------------------------------------------------------ histograma x geométrica
def test_histograma_geometrica_a_mao():
    h = histograma_geometrica([1, 1, 2, 3, 5], p=0.5, d_max=3)
    assert list(h["d"]) == [1, 2, 3]
    assert list(h["observado"]) == [2, 1, 2]  # a última faixa é "≥ 3" (3 e 5)
    np.testing.assert_allclose(h["esperado"], 5 * np.array([0.5, 0.25, 0.25]))
    assert h["observado"].sum() == 5 and h["esperado"].sum() == pytest.approx(5.0)


def test_histograma_geometrica_formula_pmf():
    p = 0.75
    h = histograma_geometrica([1] * 10, p=p, d_max=6)
    esperado = 10 * np.array([(1 - p) * p ** (d - 1) for d in range(1, 6)] + [p**5])
    np.testing.assert_allclose(h["esperado"], esperado, atol=1e-12)


def test_histograma_geometrica_sem_duracoes_levanta_erro():
    with pytest.raises(ValueError):
        histograma_geometrica([], p=0.5, d_max=3)


# ------------------------------------------------------------------ dados reais: confere com implementação independente
def _corridas_independente(serie):
    """Caminho 2: separa segmentos por pandas.diff e agrupa com itertools.groupby."""
    s = serie.copy()
    s["nivel"] = pd.to_numeric(s["nivel"], errors="coerce")
    s["data_iniSE"] = pd.to_datetime(s["data_iniSE"])
    quebra = (s["data_iniSE"].diff() != pd.Timedelta(days=7)) | s["nivel"].isna() | s["nivel"].shift().isna()
    seg_id = quebra.cumsum()
    out = []
    for _, bloco in s[s["nivel"].notna()].groupby(seg_id[s["nivel"].notna()]):
        runs = [(k, len(list(g))) for k, g in itertools.groupby(bloco["nivel"].astype(int))]
        for idx, (k, n) in enumerate(runs):
            out.append((k, n, idx == 0 or idx == len(runs) - 1))
    return out


@pytest.mark.parametrize("geocode", [SERRA, ESCOLHIDO])
def test_corridas_reais_conferem_com_implementacao_independente(geocode):
    serie = carregar_serie(geocode)
    c = corridas(serie)
    esperado = _corridas_independente(serie)
    obtido = list(zip(c["nivel"].astype(int), c["duracao"].astype(int), c["censurada"].astype(bool)))
    assert obtido == esperado


@pytest.mark.parametrize("geocode", [SERRA, ESCOLHIDO])
def test_semanas_nas_corridas_igualam_ocupacao_da_etapa3(geocode):
    _, ocup = contagens_e_ocupacao([geocode])
    c = corridas(carregar_serie(geocode))
    por_nivel = c.groupby("nivel")["duracao"].sum().reindex([1, 2, 3, 4]).fillna(0).astype(int).to_numpy()
    np.testing.assert_array_equal(por_nivel, ocup)


def _passagem_independente(serie, i, j):
    """Caminho 2 (força bruta): segmentos por diff de datas; para cada visita a i, varre à frente até achar j."""
    s = serie.copy()
    s["nivel"] = pd.to_numeric(s["nivel"], errors="coerce")
    quebra = (pd.to_datetime(s["data_iniSE"]).diff() != pd.Timedelta(days=7)) | s["nivel"].isna() | s["nivel"].shift().isna()
    tempos, cens = [], 0
    for _, bloco in s[s["nivel"].notna()].groupby(quebra.cumsum()[s["nivel"].notna()]):
        L = bloco["nivel"].astype(int).tolist()
        for t, x in enumerate(L):
            if x != i:
                continue
            achou = next((u - t for u in range(t + 1, len(L)) if L[u] == j), None)
            if achou is None:
                cens += 1
            else:
                tempos.append(achou)
    return sorted(tempos), cens


@pytest.mark.parametrize("geocode", [SERRA, ESCOLHIDO])
@pytest.mark.parametrize("i,j", [(1, 4), (4, 1), (1, 2), (3, 1)])
def test_primeira_passagem_real_confere_com_forca_bruta(geocode, i, j):
    serie = carregar_serie(geocode)
    t, cens = tempos_primeira_passagem(serie, i, j)
    esperado, cens_esp = _passagem_independente(serie, i, j)
    assert sorted(t.tolist()) == esperado and cens == cens_esp


def test_media_das_duracoes_sem_censura_e_igual_a_permanencia_do_modelo_quando_nao_ha_censura():
    """Identidade: se todas as corridas do nível i terminam dentro dos dados, a duração média = n_i/(n_i − n_ii) = 1/(1 − p̂_ii)."""
    s2 = _serie([2, 1, 1, 2, 1, 1, 1, 2, 3, 1, 2])  # corridas de 1 internas: durações 2, 3, 1
    from src.dados import contar_transicoes

    C = contar_transicoes(s2)
    p11 = C[0, 0] / C[0].sum()
    d = duracoes_sem_censura(s2)[1]
    assert sorted(d) == [1, 2, 3]
    assert d.mean() == pytest.approx(1.0 / (1.0 - p11), abs=1e-12)


# ------------------------------------------------------------------ analisar e pipeline
P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])


def test_permanencia_modelo_a_mao():
    pm = etapa5.permanencia_media(P_REF)
    np.testing.assert_allclose(pm, 1.0 / (1.0 - np.diag(P_REF)), atol=1e-12)
    np.testing.assert_allclose(pm, [10.0, 1 / 0.35, 1 / 0.35, 4.0], atol=1e-12)


def test_permanencia_modelo_rejeita_estado_absorvente():
    P = np.eye(2)
    with pytest.raises(ValueError):
        etapa5.permanencia_media(P)


def test_analisar_matriz_reproduz_tabela1():
    r = etapa5.analisar_matriz(P_REF)
    np.testing.assert_allclose(np.diag(r["M_sistemas"]), [2.004128, 4.855455, 5.743011, 8.267802], atol=1e-6)
    assert r["m14"] == pytest.approx(46.873065, abs=1e-6)
    assert r["m41"] == pytest.approx(17.035647, abs=1e-6)
    assert r["kac_max"] < 1e-10 and r["dif_sistemas_fundamental"] < 1e-10


def _P(geocodes):
    C, _ = contagens_e_ocupacao(geocodes_do_estado() if geocodes == "estado" else geocodes)
    return matriz_transicao(C)


TERRITORIOS = [[SERRA], [ESCOLHIDO], "estado"]


@pytest.mark.parametrize("geocodes", TERRITORIOS, ids=["serra", "vitoria", "es"])
def test_verificacoes_obrigatorias_nos_dados_reais(geocodes):
    r = etapa5.analisar_matriz(_P(geocodes))
    assert r["kac_max"] < 1e-10
    assert r["dif_sistemas_fundamental"] < 1e-10
    np.testing.assert_allclose(np.diag(r["M_sistemas"]), 1.0 / r["pi"], atol=1e-10)


def test_main_grava_tabelas_e_figuras(tmp_path):
    etapa5.main(saida=tmp_path)
    tab = tmp_path / "tabelas"
    for nome in ("e5_permanencia.csv", "e5_histograma_nivel4.csv", "e5_M.csv", "e5_resumo.csv"):
        assert (tab / nome).stat().st_size > 0, nome
    for nome in ("e5_hist_nivel4_serra.png", "e5_hist_nivel4_vitoria.png", "e5_hist_nivel4_es.png"):
        assert (tmp_path / "figuras" / nome).stat().st_size > 1000, nome
    resumo = pd.read_csv(tab / "e5_resumo.csv")
    assert set(resumo["territorio"]) == {"Serra", "Vitória", "ES"}
    assert (resumo["kac_max"] < 1e-10).all()
    perm = pd.read_csv(tab / "e5_permanencia.csv")
    assert len(perm) == 12 and set(perm["nivel"]) == {1, 2, 3, 4}


def test_main_para_se_kac_falhar(tmp_path, monkeypatch):
    original = etapa5.analisar_matriz

    def adulterada(P, pi=None):
        r = original(P, pi)
        r["kac_max"] = 1e-3
        return r

    monkeypatch.setattr(etapa5, "analisar_matriz", adulterada)
    with pytest.raises(SystemExit):
        etapa5.main(saida=tmp_path)


def test_main_para_se_os_dois_caminhos_de_M_discordam(tmp_path, monkeypatch):
    original = etapa5.analisar_matriz

    def adulterada(P, pi=None):
        r = original(P, pi)
        r["dif_sistemas_fundamental"] = 1e-4
        return r

    monkeypatch.setattr(etapa5, "analisar_matriz", adulterada)
    with pytest.raises(SystemExit):
        etapa5.main(saida=tmp_path)


def test_verificacoes_detectam_pi_errado():
    """kac_max e dif_sistemas_fundamental não são constantes: com π adulterado, os dois disparam."""
    pi = np.array([0.4, 0.2, 0.2, 0.2])  # não é a estacionária de P_REF
    r = etapa5.analisar_matriz(P_REF, pi=pi)
    assert r["kac_max"] > 1e-2
    assert r["dif_sistemas_fundamental"] > 1e-2


def test_contagem_de_censuradas_e_de_validas_fecha_com_o_total_de_corridas_na_serra():
    e = etapa5.dados_empiricos([SERRA])
    c = corridas(carregar_serie(SERRA))
    assert sum(e["censuradas"].values()) == int(c["censurada"].sum()) > 0
    assert sum(len(v) for v in e["duracoes"].values()) == int((~c["censurada"]).sum())


def test_pipeline_do_es_usa_os_78_municipios(tmp_path):
    etapa5.main(saida=tmp_path)
    perm = pd.read_csv(tmp_path / "tabelas" / "e5_permanencia.csv")
    perm = perm[perm["territorio"] == "ES"].set_index("nivel")
    total = {n: 0 for n in range(1, 5)}
    for g in geocodes_do_estado():
        c = corridas(carregar_serie(g))
        for n in range(1, 5):
            total[n] += int((c["nivel"] == n).sum())
    for n in range(1, 5):
        assert perm.loc[n, "n_corridas_validas"] + perm.loc[n, "n_corridas_censuradas"] == total[n]
# [/IA]
