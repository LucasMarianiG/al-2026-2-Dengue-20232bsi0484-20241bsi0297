# [IA] Claude Sonnet 5.5 - 2026-11-23 - tests/test_etapa6.py: testes do pipeline da Etapa 6 (episodios 1->2, IC binomial, analise do modelo, verificacoes e execucao principal)
"""Testes de src/etapa6.py: episódios nos dados, IC binomial, análise do modelo e execução principal."""
import numpy as np
import pandas as pd
import pytest

from src import etapa6
from src.etapa3 import ESCOLHIDO, SERRA, contagens_e_ocupacao, geocodes_do_estado, matriz_transicao
from src.etapa5 import _segmentos
from src.etapa6 import (
    analise_B,
    analise_vermelho,
    contar_episodios,
    episodios_laco,
    episodios_regex,
    ic_binomial,
    ic_wilson,
    linha_episodios,
    segmentos_do_territorio,
    tabela_B,
    tabela_vermelho,
    verificar,
)

P_REF = np.array([
    [0.90, 0.08, 0.02, 0.00],
    [0.20, 0.65, 0.13, 0.02],
    [0.05, 0.15, 0.65, 0.15],
    [0.00, 0.05, 0.20, 0.75],
])
T_TABELA1 = np.array([46.873065, 39.473684, 26.470588])
VAR_TABELA1 = np.array([1788.363542, 1732.320352, 1427.271899])
B_TABELA1 = np.array([[0.742718, 0.257282], [0.461165, 0.538835]])


def _serie(niveis, lacunas=()):
    """Série semanal sintética: semanas consecutivas, saltando uma semana depois dos índices em `lacunas`."""
    datas, d = [], pd.Timestamp("2020-01-05")
    for k in range(len(niveis)):
        datas.append(d)
        d += pd.Timedelta(days=14 if k in lacunas else 7)
    return pd.DataFrame({"data_iniSE": datas, "nivel": niveis})


# (sequência de níveis, (terminados em 1, terminados em 4, censurados)), conferidos à mão
CASOS_EPISODIOS = [
    ([1, 2, 3, 4], (0, 1, 0)),             # entra no 2 vindo do 1 e chega ao vermelho
    ([1, 2, 1], (1, 0, 0)),                # volta ao verde
    ([1, 2, 2, 3, 2], (0, 0, 1)),          # a série acaba antes de 1 ou 4: censurado
    ([2, 2, 1], (0, 0, 0)),                # não entrou vindo do 1
    ([2, 4], (0, 0, 0)),                   # começa no 2 sem semana anterior
    ([3, 2, 4], (0, 0, 0)),                # entrou no 2 vindo do 3, não do 1
    ([1, 2, 1, 2, 4], (1, 1, 0)),          # o 1 que encerra o 1º episódio é a entrada do 2º
    ([1, 2, 1, 2, 1], (2, 0, 0)),
    ([1, 2, 3, 2, 4], (0, 1, 0)),          # voltar ao 2 dentro do episódio não abre outro
    ([1, 1, 2, 2, 4, 1, 2, 3, 3, 1], (1, 1, 0)),
    ([1, 2], (0, 0, 1)),
    ([1], (0, 0, 0)),
    ([], (0, 0, 0)),
]


# ---------------------------------------------------------------- episódios
@pytest.mark.parametrize("sequencia,esperado", CASOS_EPISODIOS)
def test_episodios_a_mao_pelos_dois_caminhos(sequencia, esperado):
    assert tuple(episodios_laco(sequencia)) == esperado
    assert tuple(episodios_regex(sequencia)) == esperado


def test_laco_e_regex_concordam_em_sequencias_aleatorias():
    rng = np.random.default_rng(2026)
    for _ in range(400):
        L = rng.integers(1, 5, size=int(rng.integers(0, 45))).tolist()
        assert episodios_laco(L) == episodios_regex(L), L


def test_cada_entrada_1_para_2_origina_exatamente_um_episodio():
    rng = np.random.default_rng(2026)
    for _ in range(100):
        L = rng.integers(1, 5, size=40).tolist()
        entradas = sum(1 for t in range(1, len(L)) if L[t - 1] == 1 and L[t] == 2)
        assert sum(episodios_laco(L)) == entradas


def test_lacuna_interrompe_o_episodio_e_o_deixa_censurado():
    # sem lacuna, 1,2,3,2,4 é um episódio terminado em 4
    assert contar_episodios(_segmentos(_serie([1, 2, 3, 2, 4]))) == {"termina_1": 0, "termina_4": 1, "censurados": 0}
    # com lacuna depois do 3: o episódio fica censurado e o 2 seguinte não tem "1" antes (nada atravessa a lacuna)
    r = contar_episodios(_segmentos(_serie([1, 2, 3, 2, 4], lacunas=(2,))))
    assert r == {"termina_1": 0, "termina_4": 0, "censurados": 1}


def test_nivel_ausente_interrompe_como_lacuna():
    r = contar_episodios(_segmentos(_serie([1, 2, np.nan, 4, 1, 2, 1])))
    assert r == {"termina_1": 1, "termina_4": 0, "censurados": 1}


def test_contar_episodios_soma_segmentos():
    segmentos = [np.array([1, 2, 4]), np.array([1, 2, 1]), np.array([1, 2])]
    assert contar_episodios(segmentos) == {"termina_1": 1, "termina_4": 1, "censurados": 1}


def test_contar_episodios_levanta_se_os_dois_caminhos_discordam(monkeypatch):
    monkeypatch.setattr(etapa6, "episodios_regex", lambda L: (9, 9, 9))
    with pytest.raises(ValueError, match="discordam"):
        contar_episodios([np.array([1, 2, 4])])


@pytest.mark.parametrize("geocode", [SERRA, ESCOLHIDO])
def test_episodios_mais_censurados_igual_a_n12_dos_dados_reais(geocode):
    C, _ = contagens_e_ocupacao([geocode])
    e = contar_episodios(segmentos_do_territorio([geocode]))
    # n_12 é a contagem de transições 1→2: C[0, 1] em índices 0..3
    assert e["termina_1"] + e["termina_4"] + e["censurados"] == C[0, 1]


def test_episodios_mais_censurados_igual_a_n12_no_es():
    geocodes = geocodes_do_estado()
    C, _ = contagens_e_ocupacao(geocodes)
    e = contar_episodios(segmentos_do_territorio(geocodes))
    assert e["termina_1"] + e["termina_4"] + e["censurados"] == C[0, 1]


# ---------------------------------------------------------------- IC binomial
def test_wilson_valor_conhecido_5_de_10():
    inf, sup = ic_wilson(5, 10)
    assert inf == pytest.approx(0.2366, abs=1e-4)
    assert sup == pytest.approx(0.7634, abs=1e-4)


def test_wilson_extremos_ficam_em_0_1_e_n_zero_da_nan():
    assert ic_wilson(0, 20)[0] == pytest.approx(0.0, abs=1e-12)
    assert ic_wilson(20, 20)[1] == pytest.approx(1.0, abs=1e-12)
    assert all(np.isnan(ic_wilson(0, 0)))
    assert all(np.isnan(ic_binomial(0, 0)["wilson"])) and all(np.isnan(ic_binomial(0, 0)["exato"]))


@pytest.mark.parametrize("k,n", [(0, 10), (1, 12), (4, 12), (4, 11), (7, 7), (208, 783), (50, 50)])
def test_ic_contem_a_proporcao_e_wilson_confere_com_scipy(k, n):
    ic = ic_binomial(k, n)  # levanta ValueError se Wilson (à mão) diferir do scipy
    for inf, sup in (ic["wilson"], ic["exato"]):
        assert 0.0 <= inf <= k / n <= sup <= 1.0


def test_exato_valores_conhecidos_nos_extremos():
    # Clopper–Pearson: 0 de n → [0, 1 − 0,025^(1/n)]; n de n → [0,025^(1/n), 1]
    inf, sup = ic_binomial(0, 10)["exato"]
    assert inf == 0.0 and sup == pytest.approx(1 - 0.025 ** 0.1, abs=1e-9)
    inf, sup = ic_binomial(10, 10)["exato"]
    assert sup == 1.0 and inf == pytest.approx(0.025 ** 0.1, abs=1e-9)


def test_ic_estreita_com_mais_dados():
    larg = lambda k, n: np.subtract(*ic_binomial(k, n)["exato"][::-1])
    assert larg(4, 12) > larg(40, 120) > larg(400, 1200)


def test_ic_binomial_levanta_se_wilson_a_mao_diverge_do_scipy(monkeypatch):
    monkeypatch.setattr(etapa6, "ic_wilson", lambda k, n, z=1.96: (0.0, 1.0))
    with pytest.raises(ValueError, match="Wilson"):
        ic_binomial(4, 12)


# ---------------------------------------------------------------- análise do modelo (matriz de referência)
@pytest.fixture(scope="module")
def r_pref():
    rng = np.random.default_rng(2026)
    return {"nome": "Pref", "a": analise_vermelho(P_REF, rng), "b": analise_B(P_REF, rng)}


def test_analise_vermelho_reproduz_tabela1(r_pref):
    a = r_pref["a"]
    assert a["transientes"] == [0, 1, 2]
    np.testing.assert_allclose(a["t"], T_TABELA1, atol=1e-6)
    np.testing.assert_allclose(a["t_sistema"], T_TABELA1, atol=1e-6)
    np.testing.assert_allclose(a["var"], VAR_TABELA1, atol=1e-6)
    np.testing.assert_allclose(a["dp"], np.sqrt(VAR_TABELA1), atol=1e-6)
    assert a["disc_N"] < 1e-10 and a["disc_t"] < 1e-10
    assert a["m_Qm"] == 2 and a["K_serie"] > 1


def test_analise_vermelho_simulacao_perto_do_modelo(r_pref):
    a = r_pref["a"]
    np.testing.assert_allclose(a["t_sim"], T_TABELA1, atol=0.8)
    np.testing.assert_allclose(a["dp_sim"], np.sqrt(VAR_TABELA1), rtol=0.05)


def test_analise_B_reproduz_tabela1_e_B1_igual_a_1(r_pref):
    b = r_pref["b"]
    assert b["transientes"] == [1, 2] and b["absorventes"] == [0, 3]
    np.testing.assert_allclose(b["B"], B_TABELA1, atol=1e-6)
    np.testing.assert_allclose(b["B_sistema"], B_TABELA1, atol=1e-6)
    assert b["erro_B1"] < 1e-12 and b["disc_B"] < 1e-10 and b["disc_N"] < 1e-10
    np.testing.assert_allclose(b["B_sim_vermelho"], B_TABELA1[:, 1], atol=0.01)


def test_tabelas_de_vermelho_e_B(r_pref):
    tv, tb = tabela_vermelho(r_pref), tabela_B(r_pref)
    assert list(tv["nivel_inicial"]) == [1, 2, 3] and list(tb["nivel_inicial"]) == [2, 3]
    np.testing.assert_allclose(tv["t_inversa"], T_TABELA1, atol=1e-6)
    np.testing.assert_allclose(tb["soma_linha"], 1.0, atol=1e-12)
    np.testing.assert_allclose(tb["B_verde"] + tb["B_vermelho"], 1.0, atol=1e-12)
    assert (tv["disc_inversa_sistema"] < 1e-10).all()


# ---------------------------------------------------------------- verificações obrigatórias
def test_verificar_aceita_resultado_consistente(r_pref):
    verificar(r_pref)


@pytest.mark.parametrize("onde,chave,rotulo", [
    ("b", "erro_B1", "B1"), ("a", "disc_N", "Neumann"), ("b", "disc_N", "Neumann"),
    ("a", "disc_t", "t"), ("b", "disc_B", "B"),
])
def test_verificar_levanta_se_alguma_discrepancia_passa_da_tolerancia(r_pref, onde, chave, rotulo):
    adulterado = {"nome": "X", "a": dict(r_pref["a"]), "b": dict(r_pref["b"])}
    adulterado[onde][chave] = 1e-6
    with pytest.raises(ValueError, match=rotulo):
        verificar(adulterado)


# ---------------------------------------------------------------- linha de episódios
def _r_episodios(k, n, B24, censurados=1):
    return {"nome": "X", "epi": {"termina_1": n - k, "termina_4": k, "censurados": censurados}, "n_ep": n,
            "prop": k / n if n else float("nan"), "ic": ic_binomial(k, n), "B24": B24}


def test_linha_episodios_cobertura_do_IC():
    assert linha_episodios(_r_episodios(4, 12, 0.36))["B_dentro_do_IC_exato"] is True
    assert linha_episodios(_r_episodios(4, 12, 0.99))["B_dentro_do_IC_exato"] is False
    linha = linha_episodios(_r_episodios(4, 12, 0.36))
    assert linha["episodios"] == 12 and linha["terminam_em_4"] == 4 and linha["terminam_em_1"] == 8
    assert linha["exato_inf"] <= linha["proporcao_observada"] <= linha["exato_sup"]


def test_linha_episodios_sem_episodios_nao_quebra():
    linha = linha_episodios(_r_episodios(0, 0, 0.3, censurados=0))
    assert linha["episodios"] == 0 and linha["B_dentro_do_IC_exato"] is None
    assert np.isnan(linha["proporcao_observada"])


# ---------------------------------------------------------------- execução principal
@pytest.fixture(scope="module")
def saida_main(tmp_path_factory):
    pasta = tmp_path_factory.mktemp("e6")
    mp = pytest.MonkeyPatch()
    mp.setattr(etapa6, "N_SIMULACOES", 4000)
    try:
        episodios = etapa6.main(saida=pasta)
    finally:
        mp.undo()
    return pasta, episodios


def test_main_grava_tabelas_e_figuras(saida_main):
    pasta, episodios = saida_main
    for nome in ("e6_vermelho", "e6_N", "e6_B", "e6_episodios", "e6_neumann", "e6_resumo"):
        tab = pd.read_csv(pasta / "tabelas" / f"{nome}.csv")
        assert len(tab) > 0, nome
    for nome in ("e6_neumann.png", "e6_B_empirico.png"):
        assert (pasta / "figuras" / nome).stat().st_size > 1000, nome
    assert list(episodios["territorio"]) == ["Serra", "Vitória", "ES"]


def test_main_resumo_com_discrepancias_pequenas(saida_main):
    pasta, _ = saida_main
    r = pd.read_csv(pasta / "tabelas" / "e6_resumo.csv")
    for col in ("disc_N_a", "disc_N_b", "disc_t", "disc_B", "erro_B1"):
        assert (r[col] < 1e-10).all(), col
    assert (r["semente"] == 2026).all()


def test_main_episodios_fecham_com_a_matriz_de_contagens(saida_main):
    pasta, _ = saida_main
    epi = pd.read_csv(pasta / "tabelas" / "e6_episodios.csv").set_index("territorio")
    for nome, geocodes in (("Serra", [SERRA]), ("Vitória", [ESCOLHIDO]), ("ES", geocodes_do_estado())):
        C, _ = contagens_e_ocupacao(geocodes)
        assert epi.loc[nome, "episodios"] + epi.loc[nome, "censurados"] == C[0, 1], nome
        assert epi.loc[nome, "terminam_em_4"] + epi.loc[nome, "terminam_em_1"] == epi.loc[nome, "episodios"]
        assert epi.loc[nome, "proporcao_observada"] == pytest.approx(
            epi.loc[nome, "terminam_em_4"] / epi.loc[nome, "episodios"])


def test_main_t_e_B_conferem_com_calculo_independente_em_numpy_puro(saida_main):
    # Recalcula t e B com np.linalg.inv direto sobre P̂ = C/n_i, sem usar as funções de markov.py
    pasta, _ = saida_main
    tv = pd.read_csv(pasta / "tabelas" / "e6_vermelho.csv")
    tb = pd.read_csv(pasta / "tabelas" / "e6_B.csv")
    for nome, geocodes in (("Serra", [SERRA]), ("Vitória", [ESCOLHIDO]), ("ES", geocodes_do_estado())):
        C, _ = contagens_e_ocupacao(geocodes)
        P = C / C.sum(axis=1, keepdims=True)
        Q = P[:3, :3]
        t = np.linalg.inv(np.eye(3) - Q) @ np.ones(3)
        np.testing.assert_allclose(tv[tv["territorio"] == nome]["t_inversa"], t, rtol=0, atol=1e-8)
        Q2, R2 = P[1:3, 1:3], P[1:3][:, [0, 3]]
        B = np.linalg.inv(np.eye(2) - Q2) @ R2
        np.testing.assert_allclose(tb[tb["territorio"] == nome]["B_vermelho"], B[:, 1], rtol=0, atol=1e-8)


def test_main_B_do_episodio_e_o_B_do_nivel_2_para_4(saida_main):
    # a proporção observada (episódios que começam no nível 2) é comparada com B[2→4], e não com B[3→4]
    pasta, _ = saida_main
    epi = pd.read_csv(pasta / "tabelas" / "e6_episodios.csv").set_index("territorio")
    tb = pd.read_csv(pasta / "tabelas" / "e6_B.csv")
    for nome in ("Serra", "Vitória", "ES"):
        b2 = tb[(tb["territorio"] == nome) & (tb["nivel_inicial"] == 2)]["B_vermelho"].iloc[0]
        b3 = tb[(tb["territorio"] == nome) & (tb["nivel_inicial"] == 3)]["B_vermelho"].iloc[0]
        assert epi.loc[nome, "B_2_para_4"] == pytest.approx(b2, abs=1e-12)
        assert abs(b2 - b3) > 1e-3  # os dois níveis são distinguíveis, então o teste acima discrimina


def test_main_N_inversa_e_neumann_concordam(saida_main):
    pasta, _ = saida_main
    n = pd.read_csv(pasta / "tabelas" / "e6_N.csv")
    assert set(n["caso"]) == {"nivel4_absorvente", "transientes_2_3"}
    assert (n["disc"] < 1e-10).all()
    assert len(n) == 3 * (9 + 4)  # três territórios: N 3×3 e N 2×2


def test_main_erro_de_neumann_cai_com_K(saida_main):
    pasta, _ = saida_main
    t = pd.read_csv(pasta / "tabelas" / "e6_neumann.csv")
    for (nome, caso), g in t.groupby(["territorio", "caso"]):
        g = g.sort_values("K")
        assert (np.diff(g["erro_inf"].to_numpy()) <= 1e-12).all(), (nome, caso)
        assert g["erro_inf"].iloc[0] > 1.0 and g["erro_inf"].iloc[-1] < 1e-10, (nome, caso)


def test_main_e_reprodutivel_com_a_mesma_semente(saida_main, tmp_path, monkeypatch):
    pasta, _ = saida_main
    monkeypatch.setattr(etapa6, "N_SIMULACOES", 4000)
    etapa6.main(saida=tmp_path)
    for nome in ("e6_vermelho", "e6_B", "e6_episodios", "e6_resumo"):
        a = pd.read_csv(pasta / "tabelas" / f"{nome}.csv")
        b = pd.read_csv(tmp_path / "tabelas" / f"{nome}.csv")
        pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("onde,chave", [("analise_B", "erro_B1"), ("analise_B", "disc_N"), ("analise_vermelho", "disc_t")])
def test_main_para_se_uma_verificacao_obrigatoria_falhar(tmp_path, monkeypatch, onde, chave):
    original = getattr(etapa6, onde)

    def adulterada(P, rng):
        r = original(P, rng)
        r[chave] = 1e-3
        return r

    monkeypatch.setattr(etapa6, "N_SIMULACOES", 100)
    monkeypatch.setattr(etapa6, onde, adulterada)
    with pytest.raises(ValueError):
        etapa6.main(saida=tmp_path)
    assert not list((tmp_path / "tabelas").glob("e6_*.csv"))  # nada é gravado se a verificação falhar
# [/IA]
