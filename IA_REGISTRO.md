## Registro de Uso de IA - Caio Sperandio (12/11/2026)

* **Ferramenta:** Gemini (Modo Tutor / Par de Depuração)
* **Finalidade:** Estudo, estruturação de código (TDD) e depuração de ambiente.
* **Interações e Prompts:**
  1. **Resumo do escopo:** Solicitei o resumo do PDF do enunciado e do plano de trabalho em HTML para entender a divisão em duplas e as regras de entrega. *Resposta: Resumo estruturado dos entregáveis e do método Autor/Revisor.*
  2. **Estudo sobre o script de download:** Pedi ajuda para entender as perguntas de aquecimento sobre a "trava" do manifesto e o cálculo de hash. *Resposta: Explicação sobre a garantia de congelamento contra revisões da API e como o hash previne falhas por quebras de linha (`\r\n` vs `\n`).*
  3. **Geração de código (Etapa 3 e Testes):** Solicitei o esqueleto inicial do `test_markov.py` usando a Matriz de Referência (Tabela 1). Em seguida, fui orientado passo a passo para criar as funções da Etapa 3 em `src/markov.py` (Sistema Linear, Método das Potências e Autovetor à esquerda). *Resposta: Código base gerado e integrado com `pytest`.*
  4. **Depuração:** Pedi ajuda para resolver o erro `ModuleNotFoundError: No module named 'src'` no Git Bash. *Resposta: Orientação para criar arquivos `__init__.py` e rodar `python -m pytest`.*