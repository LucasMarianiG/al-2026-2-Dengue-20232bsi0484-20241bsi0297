"""Núcleo matemático das cadeias de Markov (Etapa 3): distribuição estacionária por três caminhos,
primitividade (menor k com P^k > 0) e limite de Wielandt.

Convenção: P é estocástica por linhas (cada linha soma 1) e π é vetor-linha com πP = π.
"""
# [IA] Claude Sonnet 5.5 - 2026-11-18 - Reescrita de src/markov.py: validação, potências com diagnóstico, autovetor com checagens, resíduo, menor k e Wielandt (base original do autor B com Gemini)
import warnings
from typing import NamedTuple

import numpy as np


def validar_estocastica(P, tol=1e-12):
    """Confere que P é quadrada, sem entradas negativas e com linhas somando 1. Devolve P como float."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError(f"P deve ser quadrada; formato recebido: {P.shape}")
    if (P < 0).any():
        raise ValueError("P tem entradas negativas")
    if not np.allclose(P.sum(axis=1), 1.0, atol=tol, rtol=0):
        raise ValueError("as linhas de P devem somar 1")
    return P


def calcula_distribuicao_estacionaria_sistema(P):
    """Caminho (i): resolve (Pᵀ − I)πᵀ = 0 com a última equação trocada por Σπ = 1 (LU com pivotamento).

    Se a cadeia tiver mais de uma classe recorrente, o sistema fica singular e levanta LinAlgError.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    return np.linalg.solve(A, b)


class ResultadoPotencias(NamedTuple):
    pi: np.ndarray
    iteracoes: int
    convergiu: bool


def calcula_distribuicao_estacionaria_potencias(P, pi0=None, tol=1e-12, max_iter=100_000):
    """Caminho (ii): itera π(k+1) = π(k)P até ‖π(k+1) − π(k)‖₁ < tol.

    pi0 é a distribuição inicial (padrão: uniforme). Devolve (pi, iteracoes, convergiu);
    se não convergir em max_iter passos (ex.: cadeia periódica), avisa com RuntimeWarning.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    pi = np.full(n, 1.0 / n) if pi0 is None else np.asarray(pi0, dtype=float).copy()
    if pi.shape != (n,) or (pi < 0).any() or abs(pi.sum() - 1.0) > 1e-9:
        raise ValueError("pi0 deve ser distribuição de probabilidade com n entradas")
    for k in range(1, max_iter + 1):
        novo = pi @ P
        erro = np.abs(novo - pi).sum()
        pi = novo
        if erro < tol:
            return ResultadoPotencias(pi, k, True)
    warnings.warn(f"método das potências não convergiu em {max_iter} iterações (último passo: {erro:.2e})",
                  RuntimeWarning, stacklevel=2)
    return ResultadoPotencias(pi, max_iter, False)


def calcula_distribuicao_estacionaria_autovetor(P):
    """Caminho (iii): autovetor à esquerda de P para λ = 1 (autovetor de Pᵀ), normalizado para somar 1.

    Exige λ = 1 simples (cadeia com uma só classe recorrente); senão levanta ValueError.
    """
    P = validar_estocastica(P)
    autovalores, autovetores = np.linalg.eig(P.T)
    perto_de_1 = np.abs(autovalores - 1.0) < 1e-9
    if perto_de_1.sum() != 1:
        raise ValueError(f"autovalor 1 não é simples (multiplicidade encontrada: {int(perto_de_1.sum())})")
    v = autovetores[:, np.argmax(perto_de_1)]
    if np.abs(v.imag).max() > 1e-9 * max(1.0, np.abs(v.real).max()):
        raise ValueError("autovetor de λ = 1 com parte imaginária relevante")
    v = v.real
    return v / v.sum()


def residuo_estacionario(pi, P):
    """‖πP − π‖₁: quanto π falha em ser estacionária (zero para o π verdadeiro)."""
    P = np.asarray(P, dtype=float)
    pi = np.asarray(pi, dtype=float)
    return float(np.abs(pi @ P - pi).sum())


def limite_wielandt(n):
    """Limite de Wielandt: toda matriz primitiva n×n satisfaz P^k > 0 para algum k ≤ (n−1)² + 1."""
    return (n - 1) ** 2 + 1


def menor_k_positivo(P, k_max=None):
    """Menor k ≥ 1 com todas as entradas de P^k positivas, ou None se não houver até k_max.

    Só importa o padrão de zeros: usa potências booleanas (sem erro de arredondamento).
    k_max padrão = limite de Wielandt; passando dele sem sucesso, a matriz não é primitiva.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    if k_max is None:
        k_max = limite_wielandt(n)
    padrao = P > 0
    atual = padrao.copy()
    for k in range(1, k_max + 1):
        if atual.all():
            return k
        atual = (atual.astype(np.uint8) @ padrao.astype(np.uint8)) > 0
    return None
# [/IA]


# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/markov.py (E4): espectro por dois caminhos, meia-vida, potencias por tres caminhos, diagonalizacao e kappa_1
def _ordenar_por_modulo(autovalores):
    """Índices que ordenam por módulo decrescente; empates (par conjugado) com parte imaginária positiva primeiro."""
    return sorted(range(len(autovalores)),
                  key=lambda i: (-round(float(abs(autovalores[i])), 9), -float(np.imag(autovalores[i]))))


def _enxugar(v, tol=1e-12):
    """Devolve v real se a parte imaginária de todas as entradas for desprezível; senão, complexo."""
    v = np.asarray(v)
    if np.iscomplexobj(v) and np.abs(v.imag).max() <= tol:
        return v.real.copy()
    return v


def calcula_espectro(P):
    """Autovalores de P (caminho 1: np.linalg.eig, algoritmo QR sobre a forma de Hessenberg),
    ordenados por módulo decrescente. Real se não houver parte imaginária; complexo caso contrário."""
    P = validar_estocastica(P)
    lam = np.linalg.eigvals(P)
    return _enxugar(lam[_ordenar_por_modulo(lam)])


def polinomio_caracteristico(P):
    """Coeficientes [c0=1, c1, ..., cn] de det(xI − P) = x^n + c1 x^(n−1) + ... + cn (Faddeev–LeVerrier).

    Algoritmo (n passos, só produtos de matrizes e traços): M0 = 0; Mk = P M(k−1) + c(k−1) I; ck = −tr(P Mk)/k.
    Daí: c1 = −tr(P) e cn = (−1)^n det(P). Independente de np.linalg.eig e np.linalg.det.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    I = np.eye(n)
    M = np.zeros((n, n))
    c = np.zeros(n + 1)
    c[0] = 1.0
    for k in range(1, n + 1):
        M = P @ M + c[k - 1] * I
        c[k] = -np.trace(P @ M) / k
    return c


def espectro_por_polinomio(P):
    """Caminho 2 para o espectro: raízes do polinômio característico de Faddeev–LeVerrier (np.roots),
    ordenadas por módulo decrescente."""
    P = validar_estocastica(P)
    raizes = np.roots(polinomio_caracteristico(P))
    return _enxugar(raizes[_ordenar_por_modulo(raizes)])


def calcula_meia_vida(P):
    """Meia-vida ln(1/2)/ln|λ2| em semanas: passos para o erro cair à metade. +inf se |λ2| ≥ 1 (sem convergência)."""
    modulo2 = abs(calcula_espectro(P)[1])
    if modulo2 >= 1.0:
        return float("inf")
    return float(np.log(0.5) / np.log(modulo2))


def potencia_por_multiplicacao(P, n):
    """P^n por n−1 produtos sucessivos (caminho de referência, lento: serve de conferência)."""
    P = np.asarray(P, dtype=float)
    if n < 0:
        raise ValueError("n deve ser ≥ 0")
    R = np.eye(P.shape[0])
    for _ in range(n):
        R = R @ P
    return R


def potencia_quadrados_sucessivos(P, n):
    """P^n por quadrados sucessivos (exponenciação binária), escrita explicitamente.

    Lê n em binário do bit menos significativo: a cada bit 1 multiplica o resultado pela base atual;
    depois eleva a base ao quadrado (P, P², P⁴, P⁸, ...). Devolve (P^n, número de produtos de matrizes).
    """
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P deve ser quadrada")
    if n < 0 or int(n) != n:
        raise ValueError("n deve ser inteiro ≥ 0")
    n = int(n)
    resultado = None
    base = P.copy()
    produtos = 0
    while n > 0:
        if n & 1:
            if resultado is None:
                resultado = base.copy()
            else:
                resultado = resultado @ base
                produtos += 1
        n >>= 1
        if n:
            base = base @ base
            produtos += 1
    if resultado is None:
        resultado = np.eye(P.shape[0])
    return resultado, produtos


def diagonaliza(P):
    """P = V Λ V⁻¹ com autovalores por módulo decrescente. Devolve (V, lam, Vinv).

    V vem de np.linalg.eig (colunas com norma 2 igual a 1); V⁻¹ por np.linalg.solve(V, I) (fatoração LU).
    Só vale se P for diagonalizável (V invertível); caso contrário a inversa falha ou fica muito mal condicionada.
    """
    P = validar_estocastica(P)
    lam, V = np.linalg.eig(P)
    ordem = _ordenar_por_modulo(lam)
    lam, V = lam[ordem], V[:, ordem]
    Vinv = np.linalg.solve(V, np.eye(P.shape[0]))
    return V, lam, Vinv


def potencia_por_diagonalizacao(V, lam, Vinv, n):
    """P^n = V diag(λ^n) V⁻¹. Pode ser complexa (par conjugado); a parte imaginária é ruído numérico."""
    return (V * np.asarray(lam) ** n) @ Vinv


def norma_1(M):
    """Norma 1 induzida: maior soma dos módulos por coluna."""
    return float(np.abs(np.asarray(M)).sum(axis=0).max())


def numero_condicao_1(V):
    """κ1(V) = ‖V‖1 ‖V⁻¹‖1 (inversa por LU). Depende da normalização das colunas de V."""
    V = np.asarray(V)
    Vinv = np.linalg.solve(V, np.eye(V.shape[0]))
    return norma_1(V) * norma_1(Vinv)
# [/IA]


# [IA] Claude Sonnet 5.5 - 2026-11-18 - src/markov.py (E5): primeira passagem por 4 sistemas 3x3 e pela matriz fundamental Z
def primeira_passagem_sistemas(P):
    """Matriz M = (m_ij) de tempos médios de primeira passagem, por um sistema (n−1)×(n−1) por estado-alvo j.

    Condicionando no primeiro passo (equação (2) do enunciado): para i ≠ j,
        m_ij = 1 + Σ_{k≠j} p_ik m_kj   ⇔   (I − Q_j) m_·j = 1,
    em que Q_j é P sem a linha e a coluna j. Com m_·j resolvido, o tempo de retorno é
        m_jj = 1 + Σ_{k≠j} p_jk m_kj   (e, por Kac, m_jj = 1/π_j).
    Com n = 4 são quatro sistemas 3×3, resolvidos por np.linalg.solve (LU com pivotamento).
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    M = np.zeros((n, n))
    for j in range(n):
        idx = [k for k in range(n) if k != j]
        Q = P[np.ix_(idx, idx)]
        try:
            m = np.linalg.solve(np.eye(n - 1) - Q, np.ones(n - 1))
        except np.linalg.LinAlgError as erro:
            raise ValueError(f"estado {j + 1} não é alcançável a partir de todos os outros: (I − Q) singular") from erro
        M[idx, j] = m
        M[j, j] = 1.0 + P[j, idx] @ m
    return M


def matriz_fundamental_Z(P, pi=None):
    """Matriz fundamental da cadeia ergódica: Z = (I − P + 1π)⁻¹ (inversa por LU). π é calculado se omitido."""
    P = validar_estocastica(P)
    if pi is None:
        pi = calcula_distribuicao_estacionaria_sistema(P)
    pi = np.asarray(pi, dtype=float)
    n = P.shape[0]
    return np.linalg.solve(np.eye(n) - P + np.outer(np.ones(n), pi), np.eye(n))


def primeira_passagem_fundamental(P, pi=None):
    """M pela fórmula matricial: m_ij = (z_jj − z_ij)/π_j para i ≠ j, e m_jj = 1/π_j (Kac)."""
    P = validar_estocastica(P)
    if pi is None:
        pi = calcula_distribuicao_estacionaria_sistema(P)
    pi = np.asarray(pi, dtype=float)
    Z = matriz_fundamental_Z(P, pi)
    M = (np.diag(Z)[None, :] - Z) / pi[None, :]
    np.fill_diagonal(M, 1.0 / pi)
    return M
# [/IA]


# [IA] Claude Sonnet 5.5 - 2026-11-23 - src/markov.py (E6): cadeia absorvente (N, t, variancia, B), serie de Neumann e simulacao
def particao_absorvente(P, absorventes):
    """Torna absorventes os estados dados (índices 0..n−1) e devolve (transientes, absorventes, Q, R).

    Com os transientes primeiro, P = [[Q, R], [0, I]]: Q é o bloco transiente→transiente e R é o bloco
    transiente→absorvente. As linhas dos estados absorventes em P são ignoradas (viram linhas da identidade).
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    absorventes = sorted({int(a) for a in absorventes})
    if not absorventes:
        raise ValueError("é preciso ao menos um estado absorvente")
    if absorventes[0] < 0 or absorventes[-1] >= n:
        raise ValueError(f"estados absorventes fora de 0..{n - 1}: {absorventes}")
    transientes = [i for i in range(n) if i not in absorventes]
    if not transientes:
        raise ValueError("é preciso ao menos um estado transiente")
    Q = P[np.ix_(transientes, transientes)]
    R = P[np.ix_(transientes, absorventes)]
    return transientes, absorventes, Q, R


def menor_m_norma_menor_que_1(Q, m_max=1000):
    """Menor m ≥ 1 com ‖Q^m‖∞ < 1 (hipótese da proposição 4), ou None se não houver até m_max.

    ‖·‖∞ é a maior soma, por linha, dos módulos das entradas.
    """
    Q = np.asarray(Q, dtype=float)
    Qm = np.eye(Q.shape[0])
    for m in range(1, m_max + 1):
        Qm = Qm @ Q
        if np.abs(Qm).sum(axis=1).max() < 1.0:
            return m
    return None


def matriz_fundamental_inversa(Q):
    """Caminho (i): N = (I − Q)⁻¹ por np.linalg.solve (LU com pivotamento parcial) contra a identidade.

    Antes, confere a hipótese da proposição 4 (∥Q^m∥∞ < 1 para algum m), que garante I − Q invertível.
    """
    Q = np.asarray(Q, dtype=float)
    if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
        raise ValueError(f"Q deve ser quadrada; formato recebido: {Q.shape}")
    if menor_m_norma_menor_que_1(Q) is None:
        raise ValueError("nenhum m ≤ 1000 dá ‖Q^m‖∞ < 1: algum transiente não alcança um absorvente")
    k = Q.shape[0]
    return np.linalg.solve(np.eye(k) - Q, np.eye(k))


def soma_parcial_neumann(Q, K):
    """S_K = Σ_{k=0}^{K} Q^k, por laço explícito (soma dos termos Q^k, sem inversão)."""
    Q = np.asarray(Q, dtype=float)
    if int(K) != K or K < 0:
        raise ValueError("K deve ser inteiro não negativo")
    k = Q.shape[0]
    S, termo = np.eye(k), np.eye(k)
    for _ in range(int(K)):
        termo = termo @ Q
        S = S + termo
    return S


def neumann_ate_tolerancia(Q, tol=1e-14, max_termos=200_000):
    """Caminho (ii): soma Σ Q^k até ‖Q^k‖∞ < tol. Devolve (S, K) com K o último expoente somado."""
    Q = np.asarray(Q, dtype=float)
    k = Q.shape[0]
    S, termo = np.eye(k), np.eye(k)
    for K in range(1, max_termos + 1):
        termo = termo @ Q
        S = S + termo
        if np.abs(termo).sum(axis=1).max() < tol:
            return S, K
    raise ValueError(f"série de Neumann não convergiu em {max_termos} termos")


def trajetoria_neumann(Q, N, K_max):
    """‖N − S_K‖∞ para K = 0..K_max (soma incremental), para a conferência N × série."""
    Q = np.asarray(Q, dtype=float)
    k = Q.shape[0]
    S, termo = np.eye(k), np.eye(k)
    erros = [np.abs(N - S).sum(axis=1).max()]
    for _ in range(int(K_max)):
        termo = termo @ Q
        S = S + termo
        erros.append(np.abs(N - S).sum(axis=1).max())
    return np.array(erros)


def tempo_ate_absorcao(N):
    """t = N·1: número médio de passos até a absorção, a partir de cada transiente."""
    N = np.asarray(N, dtype=float)
    return N @ np.ones(N.shape[0])


def tempo_ate_absorcao_sistema(Q):
    """Caminho independente para t: resolve (I − Q)t = 1 (sem formar N)."""
    Q = np.asarray(Q, dtype=float)
    k = Q.shape[0]
    return np.linalg.solve(np.eye(k) - Q, np.ones(k))


def variancia_tempo_absorcao(N, t=None):
    """Var = (2N − I)t − t∘t (produto de Hadamard): variância do tempo até a absorção."""
    N = np.asarray(N, dtype=float)
    if t is None:
        t = tempo_ate_absorcao(N)
    return (2.0 * N - np.eye(N.shape[0])) @ t - t * t


def prob_absorcao(N, R):
    """B = N R: B[i, a] = probabilidade de ser absorvido em a partindo do transiente i (cada linha soma 1)."""
    return np.asarray(N, dtype=float) @ np.asarray(R, dtype=float)


def prob_absorcao_sistema(Q, R):
    """Caminho independente para B: resolve (I − Q)B = R (sem formar N)."""
    Q = np.asarray(Q, dtype=float)
    return np.linalg.solve(np.eye(Q.shape[0]) - Q, np.asarray(R, dtype=float))


def simular_absorcao(P, absorventes, inicial, n_trajetorias, rng, max_passos=100_000):
    """Terceiro caminho: simula n trajetórias a partir do estado `inicial` (transiente) até a absorção.

    Devolve (tempos, destinos): passos até a absorção e índice (em 0..n−1) do absorvente atingido.
    A próxima posição é sorteada pela inversa da distribuição acumulada da linha de P.
    """
    P = validar_estocastica(P)
    n = P.shape[0]
    absorventes = sorted({int(a) for a in absorventes})
    eh_abs = np.zeros(n, dtype=bool)
    eh_abs[absorventes] = True
    if eh_abs[inicial]:
        raise ValueError("o estado inicial deve ser transiente")
    acumulada = np.cumsum(P, axis=1)
    estado = np.full(n_trajetorias, int(inicial), dtype=int)
    tempo = np.zeros(n_trajetorias, dtype=int)
    ativo = np.ones(n_trajetorias, dtype=bool)
    for passo in range(1, max_passos + 1):
        idx = np.flatnonzero(ativo)
        if idx.size == 0:
            break
        u = rng.random(idx.size)
        novo = np.minimum((u[:, None] > acumulada[estado[idx]]).sum(axis=1), n - 1)
        estado[idx] = novo
        tempo[idx] = passo
        ativo[idx[eh_abs[novo]]] = False
    if ativo.any():
        raise ValueError(f"{int(ativo.sum())} trajetórias não foram absorvidas em {max_passos} passos")
    return tempo, estado
# [/IA]
