## Registro de Uso de IA - Caio Sperandio (12/11/2026)

* **Ferramenta:** Gemini (Modo Tutor / Par de Depuração)
* **Finalidade:** Estudo, estruturação de código (TDD) e depuração de ambiente.
* **Interações e Prompts:**
  1. **Resumo do escopo:** Solicitei o resumo do PDF do enunciado e do plano de trabalho em HTML para entender a divisão em duplas e as regras de entrega. *Resposta: Resumo estruturado dos entregáveis e do método Autor/Revisor.*
  2. **Estudo sobre o script de download:** Pedi ajuda para entender as perguntas de aquecimento sobre a "trava" do manifesto e o cálculo de hash. *Resposta: Explicação sobre a garantia de congelamento contra revisões da API e como o hash previne falhas por quebras de linha (`\r\n` vs `\n`).*
  3. **Geração de código (Etapa 3 e Testes):** Solicitei o esqueleto inicial do `test_markov.py` usando a Matriz de Referência (Tabela 1). Em seguida, fui orientado passo a passo para criar as funções da Etapa 3 em `src/markov.py` (Sistema Linear, Método das Potências e Autovetor à esquerda). *Resposta: Código base gerado e integrado com `pytest`.*
  4. **Depuração:** Pedi ajuda para resolver o erro `ModuleNotFoundError: No module named 'src'` no Git Bash. *Resposta: Orientação para criar arquivos `__init__.py` e rodar `python -m pytest`.*

  
---

## Registro de Uso de IA - Lucas (integrante A)

Ferramenta: Claude Sonnet 5.5 (Anthropic), via claude.ai, em um Project do trabalho, conversa
"Canal B - engenharia". Os pedidos abaixo
são descritos de forma resumida; o link traz o texto integral.

| Data | Etapa | Finalidade | O que pedi (resumo) | O que foi produzido | Aceito / modificado / rejeitado | Como verifiquei |
|---|---|---|---|---|---|---|
| 10/11/2026 | Planejamento | Estudo | Leitura do enunciado e do plano; orientação sobre repositório, papéis A/B e setup | Diagnóstico das regras, divisão autor/revisor, comandos de Git e venv | Aceito, adaptado à dupla | Conferi contra o PDF do enunciado; rodei `git status`, `git log`, `python --version` |
| 10/11/2026 | Planejamento | Estudo | Uso do plano de execução em HTML (gerado com IA, Claude, em outra conversa; modelo: CONFIRMAR) como guia de cronograma e checklist | Plano de execução | Aceito como guia, não como fonte | Cada regra conferida no enunciado |
| 11/11/2026 | E1 | Depuração | Rascunho de teste da API do IBGE e do InfoDengue, fora do repositório | Script de rascunho (descartado) | Aceito | Saída: 78 municípios, 574 linhas, colunas SE, nivel, data_iniSE, versao_modelo |
| 11/11/2026 | E1 | Código | `src/download.py` (download único, manifesto SHA-256, trava), `.gitattributes`, `.gitkeep` | `src/download.py` (marcado com `# [IA]`), `.gitattributes` | Aceito (CONFIRMAR se alterou algo) | Ensaio com 2 municípios em pasta fora do repositório; trava testada rodando duas vezes |
| 14/11/2026 | E1 | Execução | Roteiro para o download oficial e a conferência dos hashes | Comandos de conferência (PowerShell) | Aceito | 79 arquivos, 83 linhas no manifesto, 0 divergências |
| 12 a 16/11/2026 | Gestão | Revisão | Revisão do estado do repositório (commits, testes, registro de IA) | Lista de problemas e plano de correção | Aceito | Rodei `pytest` e confirmei cada ponto |
| 16/11/2026 | Arrumação | Código e documentação | `pytest.ini`, testes da E4 como skip, README | Esses três arquivos | Aceito | `pytest`: 4 passed, 2 skipped |