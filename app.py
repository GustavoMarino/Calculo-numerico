import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from scipy.integrate import simpson, trapezoid
from scipy.optimize import curve_fit

st.set_page_config(layout="wide")
st.title("Resolução de Exercícios - Cálculo Numérico com Python")

questoes = [
    "Questão 1", "Questão 2", "Questão 3", "Questão 4 e 5",
    "Questão 6", "Questões 7, 8 e 9", "Questão 10"
]

escolha = st.sidebar.radio("Escolha a questão", questoes)

if escolha == "Questão 1":
    st.header("Questão 1: Interpolação de Lagrange")
    st.markdown("""
    **Enunciado:**
    Encontre a raiz da função \(y(x)\) dada pelos pontos abaixo. Use interpolação de Lagrange sobre:
    (a) três pontos consecutivos, e (b) quatro pontos consecutivos.

    \(x = [0, 0.5, 1, 1.5, 2, 2.5, 3]\)
    
    \(y(x) = [1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727]\)

    **Explicação:**
    A interpolação de Lagrange gera um polinômio que passa exatamente pelos pontos dados. Ao usar 3 ou 4 pontos consecutivos, buscamos um polinômio que se aproxime da função e possivelmente revele raízes (valores onde y(x)=0).
    """)
    x = np.array([0, 0.5, 1, 1.5, 2, 2.5, 3])
    y = np.array([1.8421, 2.4694, 2.4921, 1.9047, 0.8509, -0.4112, -1.5727])
    poly3 = lagrange(x[2:5], y[2:5])
    poly4 = lagrange(x[2:6], y[2:6])
    x_plot = np.linspace(1, 2.5, 300)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o', label='Dados')
    ax.plot(x_plot, poly3(x_plot), label='Lagrange 3 pontos')
    ax.plot(x_plot, poly4(x_plot), label='Lagrange 4 pontos')
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

elif escolha == "Questão 2":
    st.header("Questão 2: Polinômio interpolador grau ≤ 2")
    st.markdown("""
    **Enunciado:**
    Encontre um polinômio \(p\) de grau no máximo 2 que satisfaça:
    \(p(-1) = -32\), \(p(2) = 1\), \(p(4) = 3\)

    **Explicação:**
    A interpolação com 3 pontos distintos gera um polinômio de grau 2 que passa por todos eles. Esse tipo de interpolação é útil para prever valores intermediários ou suavizar tendências.
    """)
    x = np.array([-1, 2, 4])
    y = np.array([-32, 1, 3])
    poly = lagrange(x, y)
    x_plot = np.linspace(-2, 5, 300)
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o')
    ax.plot(x_plot, poly(x_plot), label='Polinômio grau ≤ 2')
    ax.axhline(0, color='gray', linestyle='--')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

elif escolha == "Questão 3":
    st.header("Questão 3: Ajuste Exponencial por Mínimos Quadrados")
    st.markdown("""
    **Enunciado:**
    Com os mesmos dados da questão anterior, encontre um modelo exponencial \(y = ae^{bx}\) que melhor se ajuste aos dados usando o método dos mínimos quadrados.

    **Explicação:**
    O método dos mínimos quadrados é usado para encontrar os parâmetros \(a\) e \(b\) que minimizam o erro entre o modelo exponencial e os dados. Esse tipo de ajuste é útil para modelar crescimento (ou decaimento) exponencial.
    """)
    x = np.array([-1, 2, 4])
    y = np.array([-32, 1, 3])

    def modelo_exp(x, a, b):
        return a * np.exp(b * x)

    try:
        popt, _ = curve_fit(modelo_exp, x, y, maxfev=10000)
        x_plot = np.linspace(-2, 5, 300)
        fig, ax = plt.subplots()
        ax.plot(x, y, 'o')
        ax.plot(x_plot, modelo_exp(x_plot, *popt), label='Ajuste exponencial')
        ax.axhline(0, color='gray', linestyle='--')
        ax.legend()
        ax.grid()
        st.pyplot(fig)
    except:
        st.error("Erro ao ajustar modelo exponencial: dados negativos.")
    
elif escolha == "Questão 4 e 5":
    st.header("Questões 4 e 5: Integração Numérica - Trapézios e Simpson")
    st.markdown("""
    **Enunciado (Q4):**
    Seja \(f(x) = \frac{(x - 2)^2}{(x + 3)^3}\). Estime \(A = \int_0^1 f(x)\,dx\) pela regra dos trapézios e pela regra de Simpson usando os valores tabelados.

    **Enunciado (Q5):**
    Utilize a fórmula teórica do erro da regra dos trapézios para estimar o erro cometido ao calcular a integral da questão anterior.

    **Explicação:**
    Utilizamos métodos numéricos de integração para estimar áreas sob curvas. A regra dos trapézios usa aproximações lineares, enquanto Simpson usa parábolas. A questão 5 trata da estimativa do erro da aproximação por trapézios.
    """)
    x = np.array([0.00, 0.25, 0.50, 0.75, 1.00])
    y = np.array([1.4815e-1, 8.9213e-2, 5.2478e-2, 2.9630e-2, 1.5625e-2])
    trap = trapezoid(y, x)
    simp = simpson(y, x)
    st.write(f"Área pela regra dos trapézios: {trap:.6f}")
    st.write(f"Área pela regra de Simpson: {simp:.6f}")
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')
    ax.grid()
    st.pyplot(fig)

elif escolha == "Questão 6":
    st.header("Questão 6: Integração no intervalo [0, 1.75]")
    st.markdown("""
    **Enunciado:**
    Calcule novamente a integral anterior, agora no intervalo de \([0, 1.75]\). Compare os resultados da regra dos trapézios e de Simpson. Use os valores tabelados.

    **Explicação:**
    Ao expandir o intervalo de integração, verificamos como os métodos de aproximação se comportam em um domínio mais amplo com mais subdivisões.
    """)
    x = np.array([0.00, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50, 1.75])
    y = np.array([1.4815e-1, 8.9213e-2, 5.2478e-2, 2.9630e-2, 1.5625e-2, 8.2150e-3, 4.0100e-3, 2.1000e-3])
    trap = trapezoid(y, x)
    simp = simpson(y, x)
    st.write(f"Trapézios: {trap:.6f}")
    st.write(f"Simpson: {simp:.6f}")
    fig, ax = plt.subplots()
    ax.plot(x, y, 'o-')
    ax.grid()
    st.pyplot(fig)

elif escolha == "Questões 7, 8 e 9":
    st.header("Questões 7, 8 e 9: EDO com Euler e RK2")
    st.markdown("""
    **Enunciado (Q7):**
    Considere a EDO \(y' = -2xy^2\), com \(y(0) = 1\). Estime a solução numérica no intervalo \([0, 1]\) usando o método de Euler com \(h = 0.1\).

    **Enunciado (Q8):**
    Resolva novamente a EDO usando o método de Runge-Kutta de 2ª ordem (RK2).

    **Enunciado (Q9):**
    Compare graficamente as soluções obtidas por Euler e RK2 com a solução exata \(y(x) = \frac{1}{1 + x^2}\).

    **Explicação:**
    Os métodos de Euler e RK2 são métodos numéricos para resolver EDOs. Nesta comparação, avaliamos a precisão de cada um em relação à solução analítica conhecida.
    """)
    def f(x, y): return -2 * x * y**2
    h = 0.1
    x_vals = np.linspace(0, 1, 11)
    y_euler = np.zeros_like(x_vals)
    y_rk2 = np.zeros_like(x_vals)
    y_euler[0] = y_rk2[0] = 1

    for i in range(1, len(x_vals)):
        y_euler[i] = y_euler[i-1] + h * f(x_vals[i-1], y_euler[i-1])
        k1 = f(x_vals[i-1], y_rk2[i-1])
        k2 = f(x_vals[i-1] + h, y_rk2[i-1] + h * k1)
        y_rk2[i] = y_rk2[i-1] + h/2 * (k1 + k2)

    y_exact = 1 / (1 + x_vals**2)
    fig, ax = plt.subplots()
    ax.plot(x_vals, y_exact, label='Exata')
    ax.plot(x_vals, y_euler, 'o--', label='Euler')
    ax.plot(x_vals, y_rk2, 's--', label='RK2')
    ax.legend()
    ax.grid()
    st.pyplot(fig)

elif escolha == "Questão 10":
    st.header("Questão 10: Ajuste de Modelos à População")
    st.markdown("""
    **Enunciado:**
    Considere os dados experimentais de crescimento de uma população. Ajuste um modelo adequado (linear, polinomial ou exponencial) e discuta qual representa melhor os dados. Use gráficos.

    | Ano | População (milhares) |
    |-----|-----------------------|
    | 2000 | 120 |
    | 2002 | 125 |
    | 2004 | 130 |
    | 2006 | 138 |
    | 2008 | 147 |
    | 2010 | 160 |
    | 2012 | 172 |
    | 2014 | 185 |
    | 2016 | 199 |
    | 2018 | 214 |
    | 2020 | 230 |
    | 2022 | 248 |
    | 2024 | 267 |

    **Explicação:**
    Utilizamos métodos de ajuste de curvas para modelar o crescimento populacional. Compararemos modelos linear, polinomial de 2º grau e exponencial, avaliando visualmente qual se adapta melhor aos dados reais.
    """)
    anos = np.arange(2000, 2025, 2)
    pop = np.array([120, 125, 130, 138, 147, 160, 172, 185, 199, 214, 230, 248, 267])
    x = anos - 2000

    def linear(x, a, b): return a * x + b
    def poly2(x, a, b, c): return a * x**2 + b * x + c
    def expo(x, a, b): return a * np.exp(b * x)

    lin, _ = curve_fit(linear, x, pop)
    pol, _ = curve_fit(poly2, x, pop)
    exp, _ = curve_fit(expo, x, pop)

    x_plot = np.linspace(0, 24, 300)
    fig, ax = plt.subplots()
    ax.plot(anos, pop, 'o', label='Dados')
    ax.plot(x_plot + 2000, linear(x_plot, *lin), label='Linear')
    ax.plot(x_plot + 2000, poly2(x_plot, *pol), label='Polinomial')
    ax.plot(x_plot + 2000, expo(x_plot, *exp), label='Exponencial')
    ax.legend()
    ax.grid()
    st.pyplot(fig)