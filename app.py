import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st

from calculo import (METODOS, parsear_funcion, parsear_limite, preparar_problema, crear_f,
                     analisis_exacto, pasos_solucion, calcular_metodo, latex_metodo, fmt, _quad)
from graficos import (fig_region_2d, fig_solido_3d, fig_area, fig_metodo, fig_convergencia,
                      METODOS_DISCO)

st.set_page_config(page_title="Volumen de sólidos de revolución", page_icon="Integrales", layout="wide")
st.title("Volumen de sólidos de revolución")
st.caption("Método de discos y arandelas · Solución paso a paso, gráficas 2D/3D interactivas y verificación numérica")

# ------------------------------------------------------------------ entrada
with st.sidebar:
    st.header("Datos del problema")
    f_txt = st.text_input("Función f(x)", value="x", help="Ejemplos: x, x^2, sqrt(x), sin(x), exp(-x), 2x+1")
    g_txt = st.text_input("Segunda función g(x)  (opcional)", value="1/x",
                          help="Vacío = la región llega hasta el eje de giro (método de DISCOS). "
                               "Con una segunda función se usa el método de ARANDELAS. "
                               "El programa decide cuál es la exterior.")
    c1, c2 = st.columns(2)
    a_txt = c1.text_input("Límite a", value="1")
    b_txt = c2.text_input("Límite b", value="4")
    k_txt = st.text_input("Eje de giro:  y =", value="0",
                          help="0 = eje x. Otro valor (por ejemplo 2) gira alrededor de la recta y = 2.")
    st.divider()
    corte = st.checkbox("Ver corte del sólido (mostrar el interior)", value=False)
    n = st.slider("Número de discos para la verificación numérica (n)", 2, 1000, 8,
                  help="Con n ≤ 30 se dibujan los discos en 3D; con n ≤ 100, las secciones en 2D.")
    with st.expander("Sintaxis aceptada"):
        st.markdown("- Potencia: `x**2` o `x^2`\n- Producto: `2*x` o `2x`\n"
                    "- `sin, cos, tan, exp, ln, log, sqrt, abs`\n- Constantes: `pi`, `e`\n"
                    "- Límites y eje: `pi/2`, `-1.5`, `sqrt(2)`")
    with st.expander("Software utilizado"):
        st.markdown(f"- Python {platform.python_version()}\n"
                    + "\n".join(f"- {p} {version(p)}" for p in
                                ["streamlit", "sympy", "numpy", "scipy", "matplotlib", "plotly", "pandas"]))

try:
    f_expr = parsear_funcion(f_txt)
    k_sym, k = parsear_limite(k_txt)
    g_expr = parsear_funcion(g_txt) if g_txt.strip() else k_sym     # vacío -> el propio eje (discos)
    a_sym, a = parsear_limite(a_txt)
    b_sym, b = parsear_limite(b_txt)
    if a >= b:
        raise ValueError("Debe cumplirse a < b.")
    prob = preparar_problema(f_expr, g_expr, k_sym, a, b)
    Rk, rk, G = crear_f(prob["Rk"]), crear_f(prob["rk"]), crear_f(prob["integrando"])
    xs = np.linspace(a, b, 500)
    vR, vr = Rk(xs), rk(xs)
    if not (np.all(np.isfinite(vR[1:-1])) and np.all(np.isfinite(vr[1:-1]))):
        raise ValueError("Alguna función no está definida (o no es real) en todo el intervalo [a, b].")
    if not (np.isfinite(vR[[0, -1]]).all() and np.isfinite(vr[[0, -1]]).all()):
        st.warning("⚠️ Una función no está definida en un extremo del intervalo (integral impropia). "
                   "Algunos métodos numéricos pueden dar `nan` en ese punto.")
    ex = analisis_exacto(prob["integrando"], G, a_sym, b_sym, a, b)
    exacto = ex["exacto"]
except Exception as e:
    st.error(f"No se pudo procesar la entrada: {e}")
    st.stop()

metodo_nombre = "discos" if prob["disco"] else "arandelas"
if prob["intercambio"]:
    st.info("Se tomó **g(x)** como radio exterior R(x) y **f(x)** como radio interior r(x), "
            "porque g queda más lejos del eje de giro.")
if np.any(vR * vr < -1e-12):
    st.warning("⚠️ La región cruza el eje de giro (las curvas quedan de lados opuestos). "
               "El método de arandelas supone que ambas curvas están del mismo lado del eje; "
               "el resultado puede no ser el volumen geométrico.")
A_vals = G(xs)
if np.nanmin(A_vals) < -1e-9:
    st.warning("⚠️ Las curvas se cruzan dentro del intervalo: A(x) cambia de signo. "
               "La integral resta esa parte; el volumen geométrico real es ∫|A(x)|dx "
               f"≈ {fmt(_quad(lambda t: abs(G(t)), a, b, 300), 8)} u³.")

# ============================================================ SOLUCIÓN PASO A PASO
st.header(f"Solución paso a paso · método de {metodo_nombre}")
ps = pasos_solucion(prob, k_sym, a_sym, b_sym)

st.subheader("1. Identificar los radios")
st.write(f"Se gira la región entre las curvas alrededor de la recta $y={fmt(k)}$, desde $x={fmt(a)}$ hasta $x={fmt(b)}$.")
R_ltx, r_ltx = sp.latex(prob["R"]), sp.latex(prob["r"])
if k == 0:
    st.latex(rf"R(x) = {R_ltx}")
    st.latex(rf"r(x) = {r_ltx}" if not prob["disco"] else r"r(x) = 0 \quad (\text{la región toca el eje: discos})")
else:
    st.latex(rf"R(x) = {R_ltx} - ({fmt(k)}) = {sp.latex(prob['Rk'])}")
    st.latex(rf"r(x) = {r_ltx} - ({fmt(k)}) = {sp.latex(prob['rk'])}" if not prob["disco"]
             else r"r(x) = 0 \quad (\text{la región toca el eje: discos})")

if ps is not None:
    st.subheader("2. Plantear la integral")
    st.write("Aplicamos la fórmula del volumen para el método de " + metodo_nombre + ":")
    st.latex(ps["general"])
    st.latex(ps["plantea"])
    st.write("Desarrollamos el integrando:")
    st.latex(ps["simplificada"])

    st.subheader("3. Resolver la integral")
    st.write("Integramos cada término:")
    for t, Fi in ps["terminos"]:
        st.markdown(f"- La integral de $${t}$$ es $${Fi}$$".replace("$$", "$"))
    st.latex(ps["corchete"])
    st.write(f"Evaluamos en el límite superior ($x={fmt(b)}$):")
    st.latex(ps["Fb_sust"])
    st.latex(ps["Fb_val"])
    st.write(f"Evaluamos en el límite inferior ($x={fmt(a)}$):")
    st.latex(ps["Fa_sust"])
    st.latex(ps["Fa_val"])

    st.subheader("4. Restar y simplificar")
    st.latex(ps["resta"])
    st.latex(ps["resta_val"])
    st.latex(ps["final"])
    st.success(f"El volumen exacto del sólido de revolución es ${sp.latex(ps['V_sym'])}$ unidades cúbicas "
               f"(aproximadamente **{fmt(ps['V'], 4)}**).")
else:
    st.info("SymPy no encontró una antiderivada elemental para esta integral, así que no se puede mostrar el "
            "procedimiento a mano. Se usa la integración adaptativa de SciPy (`quad`) como valor de referencia.")
    st.success(f"Volumen ≈ **{fmt(exacto, 8)}** unidades cúbicas.")
st.metric("Volumen exacto", f"{fmt(exacto, 8)} u³")

# ============================================================ GRÁFICAS INTERACTIVAS
st.header("Gráficas")
st.caption("Interactivas: arrastra para girar el sólido, rueda del mouse para acercar, doble clic para reiniciar la vista.")
g2, g3 = st.columns(2)
with g2:
    st.plotly_chart(fig_region_2d(Rk, rk, a, b, k), width="stretch", key="region2d")
with g3:
    st.plotly_chart(fig_solido_3d(Rk, rk, a, b, k, corte=corte), width="stretch", key="solido3d")

# ============================================================ VERIFICACIÓN NUMÉRICA
st.header("Verificación numérica")
st.write("Se divide $[a, b]$ en $n$ discos (o arandelas) de grosor $\\Delta x$. En cada uno se calcula el área de su "
         "sección y se suman los volúmenes. Los métodos de trapecio y Simpson aproximan mejor la misma integral. "
         "Cada pestaña muestra la fórmula, la sustitución, la tabla de valores y las gráficas.")

columnas = {"R(x)": Rk} if prob["disco"] else {"R(x)": Rk, "r(x)": rk}
resultados = {}
tabs = st.tabs(METODOS)
for tab, nombre in zip(tabs, METODOS):
    with tab:
        m = calcular_metodo(G, a, b, n, nombre, columnas)
        resultados[nombre] = m
        L = latex_metodo(m, a_sym, b_sym)
        if nombre == "Simpson 1/3" and m["n"] != n:
            st.caption(f"Simpson necesita n par: se usó n = {m['n']}.")
        izq, der = st.columns(2)
        with izq:
            st.markdown("**1. Área de cada sección**")
            st.latex(L["area"])
            st.markdown("**2. Grosor de cada disco**")
            st.latex(L["delta"])
            st.markdown("**3. Fórmula del método**")
            st.latex(L["formula"])
            st.markdown("**4. Sustituir los valores**")
            st.latex(L["sustitucion"])
            st.markdown("**5. Calcular**")
            st.latex(L["suma"])
            st.latex(L["resultado"])
            err = abs(m["resultado"] - exacto)
            rel = err / abs(exacto) * 100 if exacto != 0 else float("nan")
            st.success(f"Aproximación: **{fmt(m['resultado'], 8)}** u³  ·  Exacto: {fmt(exacto, 8)}  ·  "
                       f"Error: {fmt(err)} ({fmt(rel, 4)} %)")
        with der:
            if nombre in METODOS_DISCO:
                st.plotly_chart(fig_solido_3d(Rk, rk, a, b, k, corte=corte, m=m, titulo=f"{nombre} · n = {m['n']}"),
                                width="stretch", key=f"s3d_{nombre}")
                st.plotly_chart(fig_region_2d(Rk, rk, a, b, k, m=m), width="stretch", key=f"s2d_{nombre}")
            st.pyplot(fig_metodo(m, G, a, b, exacto))
        with st.expander(f"Tabla de valores ({len(m['tabla'])} filas)"):
            st.caption("x = punto medio del disco" if nombre == "Discos (punto medio)"
                       else "x = nodo x_i  ·  R(x), r(x) = radios  ·  A(x) = π(R² − r²)")
            fmts = {c: "{:.6f}" for c in ("x", "R(x)", "r(x)", "A(x)", "Coef · A(x)") if c in m["tabla"]}
            st.dataframe(m["tabla"].style.format(fmts), width="stretch", hide_index=True)

# ============================================================ COMPARACIÓN
st.header("Comparación de métodos")
filas = []
for nombre, m in resultados.items():
    err = abs(m["resultado"] - exacto)
    filas.append({"Método": nombre, "n usado": m["n"], "Volumen aprox. (u³)": m["resultado"],
                  "Volumen exacto (u³)": exacto, "Error absoluto": err,
                  "Error relativo (%)": err / abs(exacto) * 100 if exacto != 0 else np.nan})
df = pd.DataFrame(filas)
st.dataframe(df.style.format({"Volumen aprox. (u³)": "{:.8f}", "Volumen exacto (u³)": "{:.8f}",
                              "Error absoluto": "{:.2e}", "Error relativo (%)": "{:.4f}"}),
             width="stretch", hide_index=True)
if df["Error absoluto"].notna().any():
    mejor = df.loc[df["Error absoluto"].idxmin(), "Método"]
    st.success(f"Con n = {n}, el método más preciso es **{mejor}**.")
st.pyplot(fig_convergencia(G, a, b, exacto))
st.caption("Mientras más empinada es la línea, más rápido converge el método al volumen exacto.")
st.caption("Software realizado por Leyvi Revolorio, Marcos Arevalo, Derek Jimenez, Kevin Cax.")