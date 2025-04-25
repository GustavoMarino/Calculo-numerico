import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy.linalg import lu_factor, lu_solve
from scipy.interpolate import lagrange

st.set_page_config(layout="wide")
st.title("Resolução de Exercícios - Cálculo Numérico lista 2")

# ================= FUNÇÕES USADAS ===================

def jacobi(A, b, x0, tol=1e-2, max_iter=100):
    n = len(b)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        x = x_new
    return x

def gauss_seidel(A, b, x0, tol=1e-2, max_iter=100):
    n = len(b)
    x = x0.copy()
    for _ in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i+1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, np.inf) < tol:
            break
        x = x_new
    return x

def cramer(A, b):
    detA = np.linalg.det(A)
    x = np.zeros(len(b))
    for i in range(len(b)):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA
    return x

# ============== QUESTÕES ====================

questoes = [
    "Questão 1", "Questão 2", "Questão 3",
    "Questão 4", "Questão 5", "Questão 6", "Questão 7"
]

escolha = st.sidebar.radio("Lista 2:", questoes)

if escolha == "Questão 1":
    st.header("Questão 1: Métodos Diretos")
    st.markdown("""
    **Enunciado:**  
    Descreva, fazendo um passo-a-passo explicativo (pode conter pseudo-código ou fluxograma), como resolver sistemas de equações lineares usando os métodos diretos:
    - (i) Regra de Cramer
    - (ii) Eliminação de Gauss
    - (iii) Decomposição LU

    **Explicação:**  
    Cada método tem sua vantagem. Cramer é simples mas inviável para grandes sistemas. Gauss resolve por escalonamento e LU permite reuso da decomposição.
    """)

elif escolha == "Questão 2":
    st.header("Questão 2: Resolvendo Sistema Linear")
    st.markdown("""
    **Enunciado:**  
    Resolva o seguinte sistema linear usando os três métodos explicados:
    
    \n
    \\[
    \\begin{cases}
    4x_1 − x_2 + x_3 = 8 \\\\
    2x_1 + 5x_2 + 2x_3 = 3 \\\\
    x_1 + 2x_2 + 4x_3 = 11
    \\end{cases}
    \\]

    **Explicação:**  
    Verificamos como diferentes métodos chegam à mesma solução e comparamos desempenho e facilidade de implementação.
    """, unsafe_allow_html=True)

    A = np.array([[4, -1, 1], [2, 5, 2], [1, 2, 4]])
    b = np.array([8, 3, 11])
    st.write("Solução por Cramer:", cramer(A, b))
    st.write("Solução por Eliminação de Gauss:", solve(A, b))
    st.write("Solução por Decomposição LU:", lu_solve(lu_factor(A), b))

elif escolha == "Questão 3":
    st.header("Questão 3: Métodos Iterativos - Conceitos")
    st.markdown("""
    **Enunciado:**  
    Descreva como resolver sistemas de equações lineares usando os métodos iterativos:
    - Jacobi
    - Gauss-Seidel

    **Explicação:**  
    Os métodos iterativos geram aproximações sucessivas a partir de um chute inicial, melhorando a precisão a cada passo.
    """)

elif escolha == "Questão 4":
    st.header("Questão 4: Iterativos + Convergência")
    st.markdown("""
    **Enunciado:**  
    Analise a convergência usando o critério das linhas e o critério de Sassenfeld.  
    Resolva o sistema:

    \\[
    \\begin{cases}
    5x_1 + 2x_2 + x_3 = 7 \\\\
    -x_1 + 4x_2 + 2x_3 = 3 \\\\
    2x_1 - 3x_2 + 10x_3 = -1
    \\end{cases}
    \\]
    Com \\( x_0 = (-2.4, 5.0, 0.3) \\) e \\( \\varepsilon < 10^{-2} \\)

    **Explicação:**  
    Verificamos se os métodos iterativos vão convergir, e então aplicamos Jacobi e Gauss-Seidel para comparar os resultados.
    """, unsafe_allow_html=True)

    A = np.array([[5, 2, 1], [-1, 4, 2], [2, -3, 10]])
    b = np.array([7, 3, -1])
    x0 = np.array([-2.4, 5.0, 0.3])
    st.write("Jacobi:", jacobi(A, b, x0))
    st.write("Gauss-Seidel:", gauss_seidel(A, b, x0))

elif escolha == "Questão 5":
    st.header("Questão 5: Método de Jacobi")
    st.markdown("""
    **Enunciado:**  
    Resolva com Jacobi:

    \\[
    \\begin{cases}
    10x_1 + 2x_2 + x_3 = 7 \\\\
    x_1 + 5x_2 + x_3 = -8 \\\\
    2x_1 + 3x_2 + 10x_3 = 6
    \\end{cases}
    \\]
    Com \\( x_0 = (0.7, -1.6, 0.6) \\) e \\( \\varepsilon < 10^{-2} \\)

    **Explicação:**  
    Aplicação prática do método iterativo de Jacobi em um sistema linear.
    """, unsafe_allow_html=True)

    A = np.array([[10, 2, 1], [1, 5, 1], [2, 3, 10]])
    b = np.array([7, -8, 6])
    x0 = np.array([0.7, -1.6, 0.6])
    st.write("Solução por Jacobi:", jacobi(A, b, x0))

elif escolha == "Questão 6":
    st.header("Questão 6: Gauss-Seidel")
    st.markdown("""
    **Enunciado:**  
    Resolva com Gauss-Seidel:

    \\[
    \\begin{cases}
    5x_1 + x_2 + x_3 = 5 \\\\
    3x_1 + 4x_2 + x_3 = 6 \\\\
    3x_1 + 3x_2 + 6x_3 = 0
    \\end{cases}
    \\]
    Com \\( x_0 = (0, 0, 0) \\) e \\( \\varepsilon < 10^{-2} \\)

    **Explicação:**  
    O método de Gauss-Seidel é geralmente mais eficiente que Jacobi por aproveitar os valores já atualizados.
    """, unsafe_allow_html=True)

    A = np.array([[5, 1, 1], [3, 4, 1], [3, 3, 6]])
    b = np.array([5, 6, 0])
    x0 = np.zeros(3)
    st.write("Solução por Gauss-Seidel:", gauss_seidel(A, b, x0))

elif escolha == "Questão 7":
    st.header("Questão 7: Interpolação - Grau 4")
    st.markdown("""
    **Enunciado:**  
    Determine o polinômio de grau 4 que passa pelos pontos:  
    (0, -1), (1, 1), (3, 3), (5, 2), (6, -2)

    **Explicação:**  
    A interpolação de Lagrange gera um polinômio que passa exatamente por todos os pontos dados.
    """)
    x = np.array([0, 1, 3, 5, 6])
    y = np.array([-1, 1, 3, 2, -2])
    poly = lagrange(x, y)
    x_plot = np.linspace(0, 6, 300)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='Pontos')
    ax.plot(x_plot, poly(x_plot), label='Polinômio grau 4')
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    ax.grid()
    st.pyplot(fig)
