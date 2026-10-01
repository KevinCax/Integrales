import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st

from calculo import (METODOS, parsear_funcion, parsear_limite, crear_f, analisis_exacto,
                     area_total, calcular_metodo, latex_metodo, latex_exacto, fmt, _sl)
from graficos import fig_area, fig_metodo, fig_convergencia

st.set_page_config(page_title="Integral definida · Área bajo la curva", page_icon="∫", layout="wide")
st.title("∫ Integral definida: área bajo la curva")
st.caption("Cálculo II · Solución exacta y numérica, con el procedimiento paso a paso y sus gráficas")

# ------------------------------------------------------------------ entrada
with st.sidebar:
    st.header("Datos de entrada")
    texto = st.text_input("Función f(x)", value="x**2 * sin(x)",
                          help="Ejemplos: x^2 + 3x, sin(x)/x, exp(-x^2), sqrt(x), ln(x), 1/(1+x^2)")
    c1, c2 = st.columns(2)
    a_txt = c1.text_input("Límite a", value="0")
    b_txt = c2.text_input("Límite b", value="3")
    n = st.slider("Subdivisiones (n)", 2, 1000, 10,
                  help="Con n ≤ 100 se dibujan los rectángulos, trapecios o parábolas.")
    with st.expander("Sintaxis aceptada"):
        st.markdown("- Potencia: `x**2` o `x^2`\n- Producto: `2*x` o `2x`\n"
                    "- `sin, cos, tan, exp, ln, log, sqrt, abs`\n- Constantes: `pi`, `e`\n"
                    "- Límites: `pi/2`, `-1.5`, `sqrt(2)`")
    with st.expander("Software utilizado"):
        st.markdown(f"- Python {platform.python_version()}\n"
                    + "\n".join(f"- {p} {version(p)}" for p in
                                ["streamlit", "sympy", "numpy", "scipy", "matplotlib", "pandas"]))

try:
    expr = parsear_funcion(texto)
    a_sym, a = parsear_limite(a_txt)
    b_sym, b = parsear_limite(b_txt)
    if a >= b:
        raise ValueError("Debe cumplirse a < b.")
    f = crear_f(expr)
    muestra = f(np.linspace(a, b, 500))
    if not np.all(np.isfinite(muestra)):
        st.warning("La función no está definida o es infinita en algunos puntos del intervalo "
                   "(integral impropia). Algunos métodos pueden dar `nan` en esos puntos.")
    ex = analisis_exacto(expr, f, a_sym, b_sym, a, b)
    exacto = ex["exacto"]
except Exception as e:
    st.error(f"No se pudo procesar la entrada: {e}")
    st.stop()

# --------------------------------------------------------- PASO 1: planteamiento
st.header("Paso 1 · Planteamiento")
st.latex(rf"A = \int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} {sp.latex(expr)}\,dx")
st.write(f"El área bajo la curva de $f(x)$ entre $x = {fmt(a)}$ y $x = {fmt(b)}$ es la **integral definida**. "
         "Se resolverá de dos formas: exacta (antiderivada) y numérica (aproximaciones).")

# --------------------------------------------------------- PASO 2: solución exacta
st.header("Paso 2 · Solución exacta")
col_txt, col_fig = st.columns([1, 1])
with col_txt:
    if ex["anti"] is not None:
        L = latex_exacto(expr, ex, a_sym, b_sym)
        st.markdown("**1. Encontrar la antiderivada**")
        st.latex(L["antiderivada"])
        st.markdown("**2. Aplicar el Teorema Fundamental del Cálculo**")
        st.latex(L["tfc"])
        st.markdown("**3. Evaluar en los límites**")
        st.latex(L["Fb"])
        st.latex(L["Fa"])
        st.markdown("**4. Restar**")
        st.latex(L["resultado"])
    else:
        st.info("SymPy no encontró una antiderivada elemental para esta función. "
                "Se usa la integración adaptativa de SciPy (`quad`) como valor de referencia.")
        st.latex(rf"\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} f(x)\,dx \approx \mathbf{{{fmt(exacto)}}}")
with col_fig:
    st.pyplot(fig_area(f, a, b, exacto))

total = area_total(f, a, b)
m1, m2 = st.columns(2)
m1.metric("Integral definida", fmt(exacto, 8))
m2.metric("Área geométrica total  ∫|f(x)|dx", fmt(total, 8))
if np.nanmin(muestra) < 0 < np.nanmax(muestra):
    st.info("La función cambia de signo en el intervalo: la integral definida **resta** el área que queda "
            "bajo el eje x (zona roja). El área geométrica total suma todas las regiones como positivas.")

# --------------------------------------------------------- PASO 3: solución numérica
st.header("Paso 3 · Solución numérica")
st.write("Se divide $[a, b]$ en $n$ subintervalos de ancho $\\Delta x$ y se aproxima el área con figuras "
         "simples. Cada pestaña muestra la fórmula, la sustitución, la tabla de valores y la gráfica.")

resultados = {}
tabs = st.tabs(METODOS)
for tab, nombre in zip(tabs, METODOS):
    with tab:
        m = calcular_metodo(f, a, b, n, nombre)
        resultados[nombre] = m
        L = latex_metodo(m, a_sym, b_sym)
        if nombre == "Simpson 1/3" and m["n"] != n:
            st.caption(f"Simpson necesita n par: se usó n = {m['n']}.")
        izq, der = st.columns([1, 1])
        with izq:
            st.markdown("**1. Ancho de cada subintervalo**")
            st.latex(L["delta"])
            st.markdown("**2. Fórmula del método**")
            st.latex(L["formula"])
            st.markdown("**3. Sustituir los valores**")
            st.latex(L["sustitucion"])
            st.markdown("**4. Calcular**")
            st.latex(L["suma"])
            st.latex(L["resultado"])
            err = abs(m["resultado"] - exacto)
            rel = err / abs(exacto) * 100 if exacto != 0 else float("nan")
            st.success(f"Aproximación: **{fmt(m['resultado'], 8)}**  ·  Exacto: {fmt(exacto, 8)}  ·  "
                       f"Error: {fmt(err)} ({fmt(rel, 4)} %)")
        with der:
            st.pyplot(fig_metodo(m, f, a, b, exacto))
        with st.expander(f"Tabla de valores ({len(m['tabla'])} filas)"):
            etiqueta = "x = punto medio del subintervalo i" if nombre == "Punto medio" else "x = nodo x_i"
            st.caption(etiqueta)
            st.dataframe(m["tabla"].style.format({"x": "{:.6f}", "f(x)": "{:.6f}", "Coef · f(x)": "{:.6f}"}),
                         width="stretch", hide_index=True)

# --------------------------------------------------------- PASO 4: comparación
st.header("Paso 4 · Comparación de métodos")
filas = []
for nombre, m in resultados.items():
    err = abs(m["resultado"] - exacto)
    filas.append({"Método": nombre, "n usado": m["n"], "Aproximación": m["resultado"],
                  "Valor exacto": exacto, "Error absoluto": err,
                  "Error relativo (%)": err / abs(exacto) * 100 if exacto != 0 else np.nan})
df = pd.DataFrame(filas)
st.dataframe(df.style.format({"Aproximación": "{:.8f}", "Valor exacto": "{:.8f}",
                              "Error absoluto": "{:.2e}", "Error relativo (%)": "{:.4f}"}),
             width="stretch", hide_index=True)
if df["Error absoluto"].notna().any():
    mejor = df.loc[df["Error absoluto"].idxmin(), "Método"]
    st.success(f"Con n = {n}, el método más preciso es **{mejor}**.")
st.pyplot(fig_convergencia(f, a, b, exacto))
st.caption("Mientras más empinada es la línea, más rápido converge el método. "
           "Simpson suele ser el que mejora más rápido al aumentar n.")
