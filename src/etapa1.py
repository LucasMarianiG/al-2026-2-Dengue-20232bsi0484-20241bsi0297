# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/etapa1.py: auditoria dos 78 municipios, hashes, elegibilidade e graficos da E1
"""Etapa 1: os dados contam a história que pensamos?

Gera, a partir dos CSV congelados em data/raw/ (sem internet):
  - results/e1_auditoria.csv            uma linha por município (Seção 8 do enunciado);
  - results/tabelas/e1_segundo_municipio.csv   escolha do 2º município e elegibilidade;
  - results/figuras/e1_serie_serra.png  sequência de níveis de Serra no tempo;
  - results/figuras/e1_mapa_calor.png   município x ano, fração de semanas em nível >= 3 (cinza).

As verificações obrigatórias (soma por nível, nenhuma transição atravessando lacuna, hash = manifesto)
rodam dentro de main(); se alguma falhar, o programa para com mensagem clara.

Uso:  python -m src.etapa1      (ou via run_all.py)
"""
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera arquivos sem abrir janela (necessário em run_all.py)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dados import N_NIVEIS, PASTA_RAW, carregar_serie, contar_transicoes, ler_bruto

ANO_INI, ANO_FIM = 2015, 2025
SERRA = 3205002
ESCOLHIDO = 3205309  # Vitória
MANIFESTO = Path("data/MANIFEST.txt")
MIN_SEMANAS_POR_NIVEL = 5
MIN_TRANSICOES = 400
NIVEIS = tuple(range(1, N_NIVEIS + 1))

COLUNAS = [
    "geocode", "municipio", "semanas_esperadas", "linhas_brutas", "semanas_observadas",
    "lacunas", "duplicatas", "datas_fora_do_calendario", "se_inconsistentes",
    "semanas_sem_nivel", "semanas_nivel_1", "semanas_nivel_2", "semanas_nivel_3",
    "semanas_nivel_4", "transicoes_validas", "transicoes_via_calendario",
    "n_versoes_modelo", "versao_modelo_min", "versao_modelo_max", "elegivel",
]


# ------------------------------------------------------------------ calendário
def inicio_se1(ano: int) -> pd.Timestamp:
    """Domingo que inicia a SE 1 do ano epidemiológico.

    Regra do calendário epidemiológico (semana de domingo a sábado): a SE 1 é a primeira
    semana que contém pelo menos 4 dias do ano, ou seja, a que contém o dia 4 de janeiro.
    """
    jan4 = pd.Timestamp(year=ano, month=1, day=4)
    return jan4 - pd.Timedelta(days=(jan4.dayofweek + 1) % 7)  # dayofweek: segunda=0 ... domingo=6


def semanas_esperadas(ano_ini: int = ANO_INI, ano_fim: int = ANO_FIM) -> pd.DatetimeIndex:
    """Todos os domingos de início de semana, da SE 1 de ano_ini até a última SE de ano_fim.

    Calculado só pela regra do calendário, sem olhar para os dados: é o que torna a contagem
    de lacunas uma verificação independente.
    """
    return pd.date_range(inicio_se1(ano_ini), inicio_se1(ano_fim + 1), freq="7D", inclusive="left")


def codigo_se(datas) -> np.ndarray:
    """Código AAAASS (ano epidemiológico e semana) de cada data de início de semana."""
    saida = []
    for d in pd.DatetimeIndex(datas):
        ano = d.year
        if d >= inicio_se1(ano + 1):
            ano += 1
        elif d < inicio_se1(ano):
            ano -= 1
        saida.append(ano * 100 + (d - inicio_se1(ano)).days // 7 + 1)
    return np.array(saida, dtype=int)


# ------------------------------------------------------------------- auditoria
def transicoes_via_calendario(serie: pd.DataFrame, esperadas: pd.DatetimeIndex) -> int:
    """Segundo caminho para contar transições válidas: percorre a grade do calendário.

    Conta os k tais que as semanas esperadas k e k+1 existem na série, ambas com nível.
    Não usa diferença de datas nem src.dados.contar_transicoes.
    """
    niveis = pd.to_numeric(serie["nivel"], errors="coerce")
    com_nivel = set(pd.to_datetime(serie["data_iniSE"])[niveis.notna()])
    return sum(1 for k in range(len(esperadas) - 1)
               if esperadas[k] in com_nivel and esperadas[k + 1] in com_nivel)


def eh_elegivel(linha) -> bool:
    """Critério do enunciado: pelo menos 5 semanas em cada nível e 400 transições válidas."""
    return bool(all(linha[f"semanas_nivel_{k}"] >= MIN_SEMANAS_POR_NIVEL for k in NIVEIS)
                and linha["transicoes_validas"] >= MIN_TRANSICOES)


def auditar_municipio(geocode, pasta=PASTA_RAW, esperadas=None) -> dict:
    """Auditoria de um município. Devolve um dicionário com as colunas da E1 (sem o nome)."""
    if esperadas is None:
        esperadas = semanas_esperadas()
    bruto = ler_bruto(geocode, pasta)
    serie = carregar_serie(geocode, pasta)  # uma linha por data (versão mais recente)
    niveis = pd.to_numeric(serie["nivel"], errors="coerce")
    na_grade = serie["data_iniSE"].isin(esperadas)

    linha = {
        "geocode": int(geocode),
        "semanas_esperadas": len(esperadas),
        "linhas_brutas": len(bruto),
        "semanas_observadas": len(serie),
        "lacunas": int(len(esperadas) - na_grade.sum()),
        "duplicatas": int(len(bruto) - len(serie)),
        "datas_fora_do_calendario": int((~na_grade).sum()),
        "se_inconsistentes": int((codigo_se(serie["data_iniSE"]) != serie["SE"].to_numpy()).sum()),
        "semanas_sem_nivel": int(niveis.isna().sum()),
    }
    for k in NIVEIS:
        linha[f"semanas_nivel_{k}"] = int((niveis == k).sum())
    linha["transicoes_validas"] = int(contar_transicoes(serie).sum())
    linha["transicoes_via_calendario"] = transicoes_via_calendario(serie, esperadas)
    linha["n_versoes_modelo"] = int(bruto["versao_modelo"].nunique())
    linha["versao_modelo_min"] = str(bruto["versao_modelo"].min())
    linha["versao_modelo_max"] = str(bruto["versao_modelo"].max())
    linha["elegivel"] = eh_elegivel(linha)
    return linha


def verificar_linha(linha) -> list:
    """Verificações obrigatórias da E1 para uma linha da auditoria. Lista vazia = tudo certo."""
    problemas = []
    soma_niveis = sum(linha[f"semanas_nivel_{k}"] for k in NIVEIS) + linha["semanas_sem_nivel"]
    if soma_niveis != linha["semanas_observadas"]:
        problemas.append(f"soma por nível ({soma_niveis}) != semanas observadas ({linha['semanas_observadas']})")
    na_grade = linha["semanas_observadas"] - linha["datas_fora_do_calendario"]
    if na_grade + linha["lacunas"] != linha["semanas_esperadas"]:
        problemas.append("observadas na grade + lacunas != semanas esperadas")
    if linha["linhas_brutas"] != linha["semanas_observadas"] + linha["duplicatas"]:
        problemas.append("linhas brutas != observadas + duplicatas")
    if linha["transicoes_validas"] != linha["transicoes_via_calendario"]:
        problemas.append("transições: caminho por datas != caminho pelo calendário")
    teto = max(linha["semanas_esperadas"] - 1 - linha["lacunas"], 0)
    if linha["transicoes_validas"] > teto:
        problemas.append("há mais transições do que pares sem lacuna (alguma atravessa lacuna)")
    return problemas


def verificar_hashes(manifesto=MANIFESTO, pasta=PASTA_RAW) -> list:
    """Compara o SHA-256 de cada arquivo de data/raw/ com o do manifesto. Lista vazia = tudo igual."""
    esperado = {}
    for texto in Path(manifesto).read_text(encoding="utf-8").splitlines():
        if not texto.strip() or texto.startswith("#"):
            continue
        campos = texto.split("\t")
        if len(campos) < 5:
            return [f"linha malformada no manifesto: {texto[:60]}"]
        esperado[campos[0]] = campos[-1].strip()

    problemas = []
    pasta = Path(pasta)
    for nome, hash_manifesto in sorted(esperado.items()):
        arquivo = pasta / nome
        if not arquivo.is_file():
            problemas.append(f"{nome}: listado no manifesto, mas ausente em {pasta}")
        elif hashlib.sha256(arquivo.read_bytes()).hexdigest() != hash_manifesto:
            problemas.append(f"{nome}: SHA-256 diferente do manifesto")
    for arquivo in sorted(pasta.iterdir()):
        if arquivo.is_file() and arquivo.name != ".gitkeep" and arquivo.name not in esperado:
            problemas.append(f"{arquivo.name}: existe em {pasta}, mas não consta no manifesto")
    return problemas


def auditar_todos(pasta=PASTA_RAW, esperadas=None) -> pd.DataFrame:
    """Auditoria dos 78 municípios (lista em data/raw/_municipios_ibge.csv)."""
    if esperadas is None:
        esperadas = semanas_esperadas()
    nomes = pd.read_csv(Path(pasta) / "_municipios_ibge.csv")
    linhas = []
    for geocode, nome in zip(nomes["geocode"], nomes["nome"]):
        linha = auditar_municipio(geocode, pasta, esperadas)
        linha["municipio"] = nome
        linhas.append(linha)
    return pd.DataFrame(linhas)[COLUNAS].sort_values("geocode").reset_index(drop=True)


# -------------------------------------------------------------------- gráficos
def fracao_nivel_alto_por_ano(geocodes, pasta=PASTA_RAW, nivel_min=3) -> pd.DataFrame:
    """Fração de semanas com nível >= nivel_min, por município (linhas) e ano epidemiológico (colunas).

    O denominador conta só as semanas com nível registrado. O ano é o do código SE (AAAASS).
    """
    linhas = {}
    for g in geocodes:
        s = carregar_serie(g, pasta)
        niveis = pd.to_numeric(s["nivel"], errors="coerce")
        d = pd.DataFrame({"ano": s["SE"] // 100, "alto": niveis >= nivel_min})[niveis.notna()]
        linhas[g] = d.groupby("ano")["alto"].mean()
    return pd.DataFrame(linhas).T.sort_index(axis=1)


def figura_serie(serie: pd.DataFrame, titulo: str, caminho) -> None:
    """Sequência semanal dos níveis de alerta de um município."""
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.step(serie["data_iniSE"], pd.to_numeric(serie["nivel"], errors="coerce"),
            where="post", color="black", linewidth=0.8)
    ax.set_yticks(NIVEIS)
    ax.set_yticklabels(["1 verde", "2 amarelo", "3 laranja", "4 vermelho"])
    ax.set_ylim(0.8, 4.2)
    ax.set_xlabel("início da semana epidemiológica")
    ax.set_ylabel("nível de alerta")
    ax.set_title(titulo)
    ax.grid(axis="y", color="0.85", linewidth=0.6)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


def figura_mapa_calor(matriz: pd.DataFrame, nomes: dict, caminho) -> None:
    """Mapa de calor município x ano, em escala de cinza (mais escuro = mais semanas em nível >= 3)."""
    matriz = matriz.loc[matriz.mean(axis=1).sort_values(ascending=False).index]
    cmap = plt.get_cmap("Greys").with_extremes(bad="white")  # células sem dado ficam brancas
    fig, ax = plt.subplots(figsize=(8.5, max(3.0, 0.13 * len(matriz) + 1.5)))
    im = ax.imshow(np.ma.masked_invalid(matriz.to_numpy(dtype=float)),
                   cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(matriz.shape[1]))
    ax.set_xticklabels(matriz.columns, fontsize=7)
    ax.set_yticks(range(matriz.shape[0]))
    ax.set_yticklabels([nomes.get(g, str(g)) for g in matriz.index], fontsize=5.5)
    ax.set_xlabel("ano epidemiológico")
    ax.set_title("Fração de semanas em nível 3 ou 4, por município e ano")
    fig.colorbar(im, ax=ax, label="fração de semanas", shrink=0.6)
    fig.tight_layout()
    fig.savefig(caminho, dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main(pasta_raw=PASTA_RAW, manifesto=MANIFESTO, saida=Path("results")) -> pd.DataFrame:
    saida = Path(saida)
    (saida / "tabelas").mkdir(parents=True, exist_ok=True)
    (saida / "figuras").mkdir(parents=True, exist_ok=True)

    # 1) integridade dos arquivos congelados
    problemas = verificar_hashes(manifesto, pasta_raw)
    if problemas:
        raise SystemExit("Hash diferente do manifesto:\n  " + "\n  ".join(problemas))

    # 2) auditoria + verificações obrigatórias
    df = auditar_todos(pasta_raw)
    for _, linha in df.iterrows():
        problemas = verificar_linha(linha.to_dict())
        if problemas:
            raise SystemExit(f"Verificação falhou em {linha['geocode']} ({linha['municipio']}): {problemas}")
    df.to_csv(saida / "e1_auditoria.csv", index=False, encoding="utf-8")

    # 3) escolha do 2º município e elegibilidade
    esc = df[df["geocode"] == ESCOLHIDO].iloc[0]
    pd.DataFrame([{
        "geocode": int(esc["geocode"]), "municipio": esc["municipio"],
        **{f"semanas_nivel_{k}": int(esc[f"semanas_nivel_{k}"]) for k in NIVEIS},
        "transicoes_validas": int(esc["transicoes_validas"]),
        "criterio_5_semanas_por_nivel": all(esc[f"semanas_nivel_{k}"] >= MIN_SEMANAS_POR_NIVEL for k in NIVEIS),
        "criterio_400_transicoes": bool(esc["transicoes_validas"] >= MIN_TRANSICOES),
        "elegivel": bool(esc["elegivel"]),
    }]).to_csv(saida / "tabelas" / "e1_segundo_municipio.csv", index=False, encoding="utf-8")

    # 4) gráficos
    figura_serie(carregar_serie(SERRA, pasta_raw), "Serra (3205002): nível de alerta por semana, 2015-2025",
                 saida / "figuras" / "e1_serie_serra.png")
    nomes = dict(zip(df["geocode"], df["municipio"]))
    figura_mapa_calor(fracao_nivel_alto_por_ano(df["geocode"], pasta_raw), nomes,
                      saida / "figuras" / "e1_mapa_calor.png")

    print(f"E1: {len(df)} municipios | semanas esperadas: {int(df['semanas_esperadas'].iloc[0])} | "
          f"transicoes validas (ES): {int(df['transicoes_validas'].sum())} | "
          f"elegiveis: {int(df['elegivel'].sum())} de {len(df)} | "
          f"{esc['municipio']} elegivel: {bool(esc['elegivel'])}")
    return df


if __name__ == "__main__":
    main()
# [/IA]
