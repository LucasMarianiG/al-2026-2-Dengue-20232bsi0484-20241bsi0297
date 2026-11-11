# [IA] Claude Sonnet 5.5 - 2026-11-11 - src/download.py: download unico e congelado (IBGE + InfoDengue) com manifesto SHA-256
"""Download único dos dados do InfoDengue (municípios do ES) e da lista do IBGE.

Uso oficial (UMA vez, na raiz do repositório):
    python src/download.py
Ensaio (não toca em data/):
    python src/download.py --saida C:/dev/rascunho-dengue/data_teste --apenas 3205002 3205309
"""
import argparse
import hashlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

URL_IBGE = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/32/municipios"
URL_INFODENGUE = (
    "https://info.dengue.mat.br/api/alertcity?geocode={g}&disease=dengue"
    "&format=csv&ew_start=1&ew_end=53&ey_start=2015&ey_end=2025"
)
COLUNAS_ESSENCIAIS = {"SE", "nivel", "data_iniSE", "versao_modelo"}
PAUSA_S = 1
TENTATIVAS = 3


def agora_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_arquivo(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def baixar(url: str) -> bytes:
    """GET com timeout; até TENTATIVAS tentativas. Devolve os bytes brutos."""
    ultimo_erro = None
    for tentativa in range(1, TENTATIVAS + 1):
        try:
            resposta = requests.get(url, timeout=60)
            resposta.raise_for_status()
            return resposta.content
        except requests.RequestException as erro:
            ultimo_erro = erro
            print(f"  tentativa {tentativa}/{TENTATIVAS} falhou: {erro}")
            time.sleep(5 * tentativa)
    raise RuntimeError(f"falha definitiva em {url}: {ultimo_erro}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download único e congelado dos dados.")
    ap.add_argument("--saida", default="data", help="pasta base (padrão: data)")
    ap.add_argument("--apenas", nargs="*", default=None,
                    help="geocodes a baixar (somente para ensaio)")
    args = ap.parse_args()

    base = Path(args.saida)
    raw = base / "raw"
    manifesto = base / "MANIFEST.txt"
    if manifesto.exists():
        sys.exit(f"ERRO: {manifesto} já existe. O download oficial roda uma única vez.")
    raw.mkdir(parents=True, exist_ok=True)

    inicio = agora_utc()
    registros = []  # (arquivo, data/hora UTC, url, linhas, sha256)

    # 1) Lista dos municípios do ES (IBGE)
    print("Obtendo a lista de municípios do IBGE...")
    quando = agora_utc()
    dados_ibge = json.loads(baixar(URL_IBGE))
    municipios = pd.DataFrame(
        [{"geocode": m["id"], "nome": m["nome"]} for m in dados_ibge]
    ).sort_values("geocode")
    if args.apenas is not None:
        municipios = municipios[municipios["geocode"].astype(str).isin(args.apenas)]
    if municipios.empty:
        sys.exit("ERRO: nenhum município selecionado.")
    caminho_lista = raw / "_municipios_ibge.csv"
    municipios.to_csv(caminho_lista, index=False, lineterminator="\n")
    registros.append((caminho_lista.name, quando, URL_IBGE,
                      len(municipios), sha256_arquivo(caminho_lista)))

    # 2) InfoDengue: um CSV por município, salvo exatamente como recebido
    total = len(municipios)
    for i, geocode in enumerate(municipios["geocode"], start=1):
        url = URL_INFODENGUE.format(g=geocode)
        quando = agora_utc()
        conteudo = baixar(url)
        df = pd.read_csv(io.BytesIO(conteudo))
        faltando = COLUNAS_ESSENCIAIS - set(df.columns)
        if faltando:
            sys.exit(f"ERRO: colunas ausentes em {geocode}: {sorted(faltando)}")
        destino = raw / f"{geocode}.csv"
        destino.write_bytes(conteudo)
        registros.append((destino.name, quando, url, len(df), sha256_arquivo(destino)))
        print(f"[{i}/{total}] {geocode}: {len(df)} linhas")
        time.sleep(PAUSA_S)

    # 3) Manifesto (só é gravado se tudo deu certo)
    cabecalho = [
        "# MANIFESTO DO DOWNLOAD (gerado por src/download.py)",
        f"# inicio (UTC): {inicio}",
        f"# python {sys.version.split()[0]} | pandas {pd.__version__} | requests {requests.__version__}",
        "# colunas separadas por TAB: arquivo, data/hora UTC, url, linhas, sha256",
    ]
    corpo = ["\t".join(str(campo) for campo in r) for r in registros]
    manifesto.write_text("\n".join(cabecalho + corpo) + "\n", encoding="utf-8")
    print(f"Concluído. {len(registros)} arquivos registrados em {manifesto}")


if __name__ == "__main__":
    main()
# [/IA]