import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys

st.set_page_config(layout="wide")
st.title("Resolução de Exercícios - Cálculo Numérico lista 1")

questoes = [
    "Questão 1",
    "Questão 2",
    "Questão 3",
    "Questão 4",
    "Questão 5",
    "Questão 6",
    "Questão 7",
    "Questão 8",
    "Questão 9",
]

escolha = st.sidebar.radio("Lista 1:", questoes)

if escolha == "Questão 1":
    st.header("Questão 1: Ponto flutuante, realmin, realmax, eps e erros")
    st.markdown("""
    **Parte 1:**  
    Mostre qual é o maior e o menor número flutuante que seu computador trabalha (realmax, realmin) e o epsilon da máquina (eps).

    **Parte 2:**  
    Avalie a expressão \((1 + x)^{-1} - 1 / x\) para \(x = 10^{-15}\) e \(x = 10^{15}\), e calcule os erros absolutos e relativos.
    """)

    realmax = sys.float_info.max
    realmin = sys.float_info.min
    eps = np.finfo(float).eps

    st.write(f"realmax: {realmax}")
    st.write(f"realmin: {realmin}")
    st.write(f"eps: {eps}")

    def calcular_erro(x):
        expr = (1 + x)**(-1) - 1
        esperado = -x / (1 + x)
        erro_abs = abs(expr - esperado)
        erro_rel = erro_abs / abs(esperado)
        return expr, esperado, erro_abs, erro_rel

    for x in [1e-15, 1e15]:
        expr, esperado, erro_abs, erro_rel = calcular_erro(x)
        st.markdown(f"**Para x = {x}**")
        st.write(f"Valor da expressão: {expr}")
        st.write(f"Valor esperado (referência analítica): {esperado}")
        st.write(f"Erro absoluto: {erro_abs}")
        st.write(f"Erro relativo: {erro_rel}")

elif escolha == "Questão 2":
    st.header("Questão 2: Polinômio próximo de ponto crítico")
    st.markdown("""
    Avalie a função
    \[
    f(x) = x^7 - 7x^6 + 21x^5 - 35x^4 + 35x^3 - 21x^2 + 7x - 1
    \]
    em 401 pontos no intervalo \([1 - 2×10^{-8}, 1 + 2×10^{-8}]\). Este polinômio é equivalente a \((x - 1)^7\), então deve ser muito pequeno perto de \(x = 1\), o que exige atenção à precisão numérica.
    """)

    x = np.linspace(1 - 2e-8, 1 + 2e-8, 401)
    f = lambda x: x**7 - 7*x**6 + 21*x**5 - 35*x**4 + 35*x**3 - 21*x**2 + 7*x - 1
    y = f(x)

    fig, ax = plt.subplots()
    ax.plot(x, y, label='f(x)', color='blue')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("f(x) próximo a x = 1")
    ax.grid()
    st.pyplot(fig)
elif escolha == "Questão 3":
    st.header("Questão 3: Sucessão \(I_n\)")
    st.markdown("""
    A sucessão é definida por:

    \[
    I_0 = \frac{1}{e(e - 1)}, \quad I_{n+1} = 1 - (n+1)I_n
    \]

    Essa sucessão converge para 0 à medida que \(n \to \infty\). Vamos calcular os 20 primeiros termos.
    """)

    e = np.exp(1)
    I = [1 / (e * (e - 1))]
    n_max = 20
    for n in range(n_max):
        I.append(1 - (n + 1) * I[-1])

    st.write("Termos da sequência \(I_n\):")
    st.write(I)

    fig, ax = plt.subplots()
    ax.plot(range(n_max + 1), I, marker='o', label="Iₙ")
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_title("Convergência da sequência Iₙ")
    ax.grid()
    ax.legend()
    st.pyplot(fig)

elif escolha == "Questão 4":
    st.header("Questão 4: Estimativa de π com Monte Carlo")
    st.markdown("""
    Gere \(n\) pares aleatórios \((x_k, y_k)\) no quadrado \([0, 1] \times [0, 1]\).  
    Conte quantos pontos caem dentro do círculo de raio 1. A estimativa é dada por:

    \[
    \pi_n = \frac{4m}{n}
    \]

    Vamos visualizar a convergência para diferentes valores de \(n\).
    """)

    ns = [100, 1000, 10000, 100000]
    erros = []

    for n in ns:
        x = np.random.rand(n)
        y = np.random.rand(n)
        m = np.sum(x**2 + y**2 <= 1)
        pi_est = 4 * m / n
        erro = abs(np.pi - pi_est)
        erros.append(erro)

    fig, ax = plt.subplots()
    ax.plot(ns, erros, marker='o')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title("Erro na estimativa de π vs número de pontos")
    ax.set_xlabel("n (número de pontos)")
    ax.set_ylabel("Erro")
    ax.grid()
    st.pyplot(fig)

elif escolha == "Questão 5":
    st.header("Questão 5: Série de Bailey–Borwein–Plouffe (BBP) para π")
    st.markdown("""
    A série:

    \[
    \pi = \sum_{m=0}^\infty \frac{16^{-m}}{8m+1} - \frac{2}{8m+4} - \frac{1}{8m+5} - \frac{1}{8m+6}
    \]

    pode ser usada para calcular π com alta precisão. Queremos saber quantos termos \(n\) são necessários para obter π com erro menor que \(10^{-4}\).
    """)

    def calc_pi_bbp(n):
        total = 0
        for m in range(n):
            total += (1 / 16**m) * (
                4 / (8 * m + 1) -
                2 / (8 * m + 4) -
                1 / (8 * m + 5) -
                1 / (8 * m + 6)
            )
        return total

    precisao = 1e-4
    n = 1
    while True:
        pi_aprox = calc_pi_bbp(n)
        if abs(pi_aprox - np.pi) < precisao:
            break
        n += 1

    st.write(f"Número de termos necessários para obter precisão < 1e-4: {n}")
    st.write(f"Aproximação de π: {pi_aprox:.10f}")
    st.write(f"Erro: {abs(pi_aprox - np.pi):.10f}")
elif escolha == "Questão 6":
    st.header("Questão 6: Barra deslizando no corredor - Método de Newton-Raphson")
    st.markdown("""
    Calcule \(\alpha\) com Newton-Raphson na equação:

    \[
    f(α) = \frac{\ell_2 \cos(\pi - γ - α)}{\sin^2(\pi - γ - α)} - \frac{\ell_1 \cos(α)}{\sin^2(α)} = 0
    \]

    Para: \(\ell_2 = 10\), \(\ell_1 = 8\), \(\gamma = 3π/5\).  
    A partir do \(\alpha\) obtido, calcule \(L\):

    \[
    L = \frac{\ell_2}{\sin(\pi - γ - α)} + \frac{\ell_1}{\sin(α)}
    \]
    """)

    l1 = 8
    l2 = 10
    gamma = 3 * np.pi / 5

    def f(alpha):
        return (l2 * np.cos(np.pi - gamma - alpha)) / (np.sin(np.pi - gamma - alpha)**2) - \
               (l1 * np.cos(alpha)) / (np.sin(alpha)**2)

    def df(alpha):
        term1 = (l2 * (np.sin(np.pi - gamma - alpha)**2 + 2 * np.cos(np.pi - gamma - alpha)**2)) / (np.sin(np.pi - gamma - alpha)**3)
        term2 = (l1 * (np.sin(alpha)**2 + 2 * np.cos(alpha)**2)) / (np.sin(alpha)**3)
        return term1 + term2

    alpha = 0.5
    for _ in range(10):
        alpha -= f(alpha) / df(alpha)

    L = l2 / np.sin(np.pi - gamma - alpha) + l1 / np.sin(alpha)

    st.write(f"α encontrado: {alpha:.6f} rad ({np.degrees(alpha):.4f}°)")
    st.write(f"Tamanho máximo L da barra: {L:.6f}")

elif escolha == "Questão 7":
    st.header("Questão 7: Método de ponto fixo para raiz quadrada")
    st.markdown("""
    Iteração:

    \[
    x_{k+1} = \frac{x_k((x_k)^2 + 3a)}{3(x_k)^2 + a}
    \]

    Para calcular \(\sqrt{a}\), com \(a > 0\).
    """)

    a = st.number_input("Escolha o valor de a:", min_value=0.0, value=2.0)
    x = 1.0
    iteracoes = [x]
    for _ in range(20):
        x = x * (x**2 + 3 * a) / (3 * x**2 + a)
        iteracoes.append(x)

    st.line_chart(iteracoes)
    st.write(f"Aproximação final após 20 iterações: {x:.10f}")
    st.write(f"Valor real: {np.sqrt(a):.10f}")
    st.write(f"Erro: {abs(x - np.sqrt(a)):.2e}")

elif escolha == "Questão 8":
    st.header("Questão 8: Volume de CO₂ - Bisseção, Falsa Posição e Newton")
    st.markdown("""
    Resolva:

    \[
    [p + a(N/V)^2](V - Nb) = kNT
    \]

    Com:  
    \(a = 0.401\), \(b = 42.7e-6\), \(T = 300K\), \(p = 3.5e7\), \(N = 1000\), \(k = 1.3806503e-23\)
    """)

    a = 0.401
    b = 42.7e-6
    k = 1.3806503e-23
    N = 1000
    T = 300
    p = 3.5e7

    def f(V):
        return (p + a * (N/V)**2) * (V - N * b) - k * N * T

    # Bisseção
    def bissecao(f, a, b, tol):
        for _ in range(1000):
            c = (a + b) / 2
            if abs(f(c)) < tol or (b - a)/2 < tol:
                return c
            if f(a) * f(c) < 0:
                b = c
            else:
                a = c
        return c

    V_bis = bissecao(f, 1e-5, 1e-3, 1e-12)

    # Newton-Raphson
    def df(V):
        return -2*a*N**3/(V**4)*(V - N*b) + (p + a*(N/V)**2)

    V_nr = 1e-4
    for _ in range(20):
        V_nr = V_nr - f(V_nr)/df(V_nr)

    st.write(f"Volume pelo método da bisseção: {V_bis:.12f}")
    st.write(f"Volume pelo método de Newton-Raphson: {V_nr:.12f}")

elif escolha == "Questão 9":
    st.header("Questão 9: Solução de \(f(x) = -1/x^3 - 1/x^2 = E\)")
    st.markdown("""
    Para valor negativo de \(E\), encontre \(x\) tal que:

    \[
    -\frac{1}{x^3} - \frac{1}{x^2} = E
    \]

    Usando bisseção, falsa posição e Newton-Raphson.  
    Critério de parada:  
    \[
    |x_{n+1} - x_n| \leq \text{eps} \cdot \max(1, |x_{n+1}|)
    \]
    """)

    E = st.number_input("Escolha o valor de E (negativo):", value=-0.1)
    eps = np.finfo(float).eps

    def f(x): return -1/x**3 - 1/x**2 - E
    def df(x): return 3/x**4 + 2/x**3

    # Newton-Raphson
    x0 = -2.0
    for _ in range(100):
        x1 = x0 - f(x0)/df(x0)
        if abs(x1 - x0) <= eps * max(1, abs(x1)):
            break
        x0 = x1

    st.write(f"Raiz com Newton-Raphson: {x1:.10f}")
