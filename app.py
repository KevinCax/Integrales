import platform
from importlib.metadata import version

import numpy as np
import pandas as pd
import sympy as sp
import streamlit as st

from calculo import (METODOS, parsear_funcion, parsear_limite, invertir, preparar_problema, crear_f,
                     analisis_exacto, pasos_solucion, calcular_metodo, latex_metodo, a_var, fmt, _quad)
from graficos import (fig_region_2d, fig_solido_3d, fig_metodo, fig_convergencia, METODOS_DISCO)

st.set_page_config(page_title="Volumen de sólidos de revolución", layout="wide")
st.title("Volumen de sólidos de revolución")
st.caption("Método de discos y arandelas · Procedimiento, resultado y gráficas")

HORIZ = "Horizontal: recta y = k  (el eje x es y = 0)"
VERT = "Vertical: recta x = k  (el eje y es x = 0)"

# nombre: (f, g, a, b, modo, k)
EJEMPLOS = {
    "— elige un ejemplo —": None,
    "Cono (y = x, de 0 a 3, eje x)": ("x", "", "0", "3", HORIZ, "0"),
    "Esfera (y = √(9 − x²), de −3 a 3, eje x)": ("sqrt(9-x^2)", "", "-3", "3", HORIZ, "0"),
    "Paraboloide (y = √x, de 0 a 4, eje x)": ("sqrt(x)", "", "0", "4", HORIZ, "0"),
    "Arandela (y = x y y = 1/x, de 1 a 4, eje x)": ("x", "1/x", "1", "4", HORIZ, "0"),
    "Giro alrededor de y = 2 (y = x², de 0 a 1)": ("x^2", "", "0", "1", HORIZ, "2"),
    "Eje y (y = x², para y entre 0 y 4)": ("x^2", "", "0", "4", VERT, "0"),
    "Eje y (y = x³, para y entre 0 y 8)": ("x^3", "", "0", "8", VERT, "0"),
}
DEFECTO = dict(f_txt="sqrt(x)", g_txt="", a_txt="0", b_txt="4", k_txt="0", modo=HORIZ)
for clave, val in DEFECTO.items():
    st.session_state.setdefault(clave, val)


def cargar_ejemplo():
    datos = EJEMPLOS[st.session_state["ejemplo"]]
    if datos:
        for clave, val in zip(("f_txt", "g_txt", "a_txt", "b_txt", "modo", "k_txt"), datos):
            st.session_state[clave] = val


# ------------------------------------------------------------------ entrada
with st.sidebar:
    st.header("Datos del problema")
    st.selectbox("Ejemplos listos", list(EJEMPLOS), key="ejemplo", on_change=cargar_ejemplo,
                 help="Elige uno para llenar los datos automáticamente.")
    modo = st.radio("¿Alrededor de qué eje se gira?", [HORIZ, VERT], key="modo",
                    help="Horizontal: el eje x u otra recta y = k. Vertical: el eje y u otra recta x = k.")
    vertical = modo == VERT
    v = "y" if vertical else "x"
    st.text_input("Función  y = f(x)", key="f_txt",
                  help="Ejemplos: x, x^2, sqrt(x), sin(x), e^(-x), 2x+1. Si giras alrededor de un eje vertical, "
                       "el programa despeja x por ti.")
    st.text_input("Segunda función (opcional)", key="g_txt",
                  help="Déjala vacía si la región llega hasta el eje de giro (método de DISCOS). "
                       "Si escribes otra curva, queda un hueco en el centro y se usa el método de ARANDELAS.")
    c1, c2 = st.columns(2)
    c1.text_input(f"{v} inicial (a)", key="a_txt")
    c2.text_input(f"{v} final (b)", key="b_txt")
    st.text_input("Eje de giro:  " + ("x =" if vertical else "y ="), key="k_txt",
                  help="0 = el eje de coordenadas. Otro valor (por ejemplo 2) gira alrededor de esa recta.")
    st.divider()
    corte = st.checkbox("Ver corte del sólido (mostrar el interior)", value=False)
    n = st.slider("Cantidad de discos (n)", 2, 1000, 8,
                  help="Más discos = más delgados = mejor aproximación. Con n ≤ 30 se dibujan los discos en 3D; "
                       "con n ≤ 100, las secciones en 2D.")
    with st.expander("Cómo escribir las funciones"):
        st.markdown("- Potencia: `x^2` o `x**2`\n- Producto: `2*x` o `2x`\n"
                    "- Funciones: `sin, cos, tan, exp, ln, sqrt, abs` (siempre con paréntesis: `sin(x)`)\n"
                    "- Constantes: `pi`, `e`  (por ejemplo `e^x`)\n"
                    "- Decimales con punto o coma: `1.5` o `1,5`\n"
                    "- Límites y eje: `pi/2`, `-1.5`, `sqrt(2)`")
    with st.expander("Software utilizado"):
        st.markdown(f"- Python {platform.python_version()}\n"
                    + "\n".join(f"- {p} {version(p)}" for p in
                                ["streamlit", "sympy", "numpy", "scipy", "matplotlib", "plotly", "pandas"]))

f_txt, g_txt, a_txt, b_txt, k_txt = (st.session_state[c] for c in ("f_txt", "g_txt", "a_txt", "b_txt", "k_txt"))
nota_inversion = []
try:
    f_in = parsear_funcion(f_txt)
    k_sym, k = parsear_limite(k_txt)
    a_sym, a = parsear_limite(a_txt)
    b_sym, b = parsear_limite(b_txt)
    if a >= b:
        raise ValueError(f"Debe cumplirse {v} inicial < {v} final.")
    g_in = parsear_funcion(g_txt) if g_txt.strip() else None
    if vertical:                                    # y = f(x)  ->  x = g(y)
        inv = invertir(f_in, a, b)
        nota_inversion.append(("f", inv))
        f_int = inv["expr"]
        if g_in is not None:
            inv2 = invertir(g_in, a, b)
            nota_inversion.append(("g", inv2))
            g_int = inv2["expr"]
        else:
            g_int = k_sym
    else:
        f_int, g_int = f_in, (g_in if g_in is not None else k_sym)
    prob = preparar_problema(f_int, g_int, k_sym, a, b)
    Rk, rk, G = crear_f(prob["Rk"]), crear_f(prob["rk"]), crear_f(prob["integrando"])
    xs = np.linspace(a, b, 500)
    vR, vr = Rk(xs), rk(xs)
    if not (np.all(np.isfinite(vR[1:-1])) and np.all(np.isfinite(vr[1:-1]))):
        raise ValueError("Alguna función no está definida (o no es real) en todo el intervalo. "
                         "Por ejemplo, sqrt(x) no existe para x negativo y 1/x no existe en 0.")
    if not (np.isfinite(vR[[0, -1]]).all() and np.isfinite(vr[[0, -1]]).all()):
        st.warning("Una función no está definida en un extremo del intervalo (integral impropia). "
                   "Algunos métodos numéricos pueden dar `nan` en ese punto.")
    ex = analisis_exacto(prob["integrando"], G, a_sym, b_sym, a, b)
    exacto = ex["exacto"]
except ValueError as e:
    st.error(f"No se pudo procesar la entrada: {e}")
    st.stop()
except Exception as e:  # noqa: BLE001 - cualquier otra falla se explica sin mostrar la traza
    st.error(f"Ocurrió un problema inesperado con estos datos ({type(e).__name__}). "
             "Revisa la función y el intervalo, o prueba con un ejemplo.")
    st.stop()


def L(texto):
    """LaTeX de una expresión interna (en x) mostrada con la variable correcta."""
    return a_var(sp.latex(texto), v)


metodo_nombre = "discos" if prob["disco"] else "arandelas"
eje_txt = f"x = {fmt(k)}" if vertical else f"y = {fmt(k)}"

# =========================================================================== 1. PLANTEAMIENTO
st.header(f"1 · Planteamiento · método de {metodo_nombre}")
if vertical:
    st.markdown(f"Curva: $y = {sp.latex(f_in)}$. Como el eje es vertical, se despeja $x$ en función de $y$:")
    for nombre, inv in nota_inversion:
        st.latex(rf"x = {inv['latex']}")
    if any(inv["ramas"] > 1 for _, inv in nota_inversion):
        st.caption("Había más de una solución: se usó la rama positiva/principal.")
if prob["intercambio"]:
    st.info("Radio exterior R = segunda función; radio interior r = primera (la segunda queda más lejos del eje).")

st.markdown(rf"Eje de giro: ${eje_txt}$  ·  Intervalo: ${v}\in[{fmt(a)},\,{fmt(b)}]$")
R_ltx, r_ltx = L(prob["R"]), L(prob["r"])
RkL, rkL = L(prob["Rk"]), L(prob["rk"])
sin_hueco = rf"r({v}) = 0"
if k == 0:
    st.latex(rf"R({v}) = {R_ltx}")
    st.latex(rf"r({v}) = {r_ltx}" if not prob["disco"] else sin_hueco)
else:
    st.latex(rf"R({v}) = {R_ltx} - ({fmt(k)}) = {RkL}")
    st.latex(rf"r({v}) = {r_ltx} - ({fmt(k)}) = {rkL}" if not prob["disco"] else sin_hueco)

if np.any(vR * vr < -1e-12):
    st.warning("La región cruza el eje de giro (las curvas quedan de lados opuestos). "
               "El método de arandelas supone que ambas curvas están del mismo lado del eje; "
               "el resultado puede no ser el volumen geométrico.")
A_vals = G(xs)
if np.nanmin(A_vals) < -1e-9:
    st.warning("Las curvas se cruzan dentro del intervalo: A cambia de signo. "
               "La integral resta esa parte; el volumen geométrico real es ∫|A|d" + v +
               f" ≈ {fmt(_quad(lambda t: abs(G(t)), a, b, 300), 8)} u³.")

g2, g3 = st.columns(2)
g2.plotly_chart(fig_region_2d(Rk, rk, a, b, k, vertical=vertical), width="stretch", key="region2d")
g3.plotly_chart(fig_solido_3d(Rk, rk, a, b, k, corte=corte, vertical=vertical), width="stretch", key="solido3d")

# =========================================================================== 2. NUMÉRICO
st.header(f"2 · Solución numérica con n = {n} discos")

columnas = {"R": Rk} if prob["disco"] else {"R": Rk, "r": rk}
columnas = {f"{nom}({v})": fun for nom, fun in columnas.items()}
resultados = {m_: calcular_metodo(G, a, b, n, m_, columnas) for m_ in METODOS}

cols_m = st.columns(len(METODOS_DISCO) + 1)
for c, nombre in zip(cols_m, METODOS_DISCO):
    c.metric(nombre, f"{fmt(resultados[nombre]['resultado'], 6)} u³")
cols_m[-1].metric("Valor exacto (referencia)", f"{fmt(exacto, 6)} u³")

tabs = st.tabs(METODOS)
for tab, nombre in zip(tabs, METODOS):
    with tab:
        m = resultados[nombre]
        Ltx = latex_metodo(m, a_sym, b_sym, v)
        if nombre == "Simpson 1/3" and m["n"] != n:
            st.caption(f"Simpson necesita n par: se usó n = {m['n']}.")
        izq, der = st.columns(2)
        with izq:
            st.markdown("**1. Área de cada sección**")
            st.latex(Ltx["area"])
            st.markdown("**2. Grosor de cada disco**")
            st.latex(Ltx["delta"])
            st.markdown("**3. Fórmula del método**")
            st.latex(Ltx["formula"])
            st.markdown("**4. Sustituir los valores**")
            st.latex(Ltx["sustitucion"])
            st.markdown("**5. Calcular**")
            st.latex(Ltx["suma"])
            st.latex(Ltx["resultado"])
            err = abs(m["resultado"] - exacto)
            rel = err / abs(exacto) * 100 if exacto != 0 else float("nan")
            st.success(f"Volumen aproximado: **{fmt(m['resultado'], 8)}** u³  ·  Exacto: {fmt(exacto, 8)}  ·  "
                       f"Error: {fmt(err)} ({fmt(rel, 4)} %)")
        with der:
            if nombre in METODOS_DISCO:
                st.plotly_chart(fig_solido_3d(Rk, rk, a, b, k, corte=corte, m=m, vertical=vertical,
                                              titulo=f"{nombre} · n = {m['n']}"),
                                width="stretch", key=f"s3d_{nombre}")
                st.plotly_chart(fig_region_2d(Rk, rk, a, b, k, m=m, vertical=vertical),
                                width="stretch", key=f"s2d_{nombre}")
            st.pyplot(fig_metodo(m, G, a, b, exacto, var=v))
        with st.expander(f"Tabla de valores ({len(m['tabla'])} filas)"):
            tabla = m["tabla"].copy()
            if nombre in METODOS_DISCO:
                tabla["Volumen del disco"] = tabla[f"A(x)"] * m["h"]
            tabla = tabla.rename(columns={"x": v, "A(x)": f"A({v})", "Coef · A(x)": f"Coef · A({v})"})
            fmts = {c: "{:.6f}" for c in tabla.columns if c not in ("i", "Coef.")}
            st.dataframe(tabla.style.format(fmts), width="stretch", hide_index=True)

# =========================================================================== 3. EXACTO
st.header("3 · Valor exacto con la integral")
ps = pasos_solucion(prob, k_sym, a_sym, b_sym, v)
if ps is not None:
    st.subheader("Paso 1 · Plantear la integral")
    st.latex(ps["general"])
    st.latex(ps["plantea"])
    st.latex(ps["simplificada"])

    st.subheader("Paso 2 · Encontrar la antiderivada")
    for t, Fi in ps["terminos"]:
        st.markdown(f"- La integral de ${t}$ es ${Fi}$")
    st.latex(ps["corchete"])

    st.subheader("Paso 3 · Evaluar en los límites")
    st.write(f"Evaluamos en el límite superior (${v}={fmt(b)}$):")
    st.latex(ps["Fb_sust"])
    st.latex(ps["Fb_val"])
    st.write(f"Evaluamos en el límite inferior (${v}={fmt(a)}$):")
    st.latex(ps["Fa_sust"])
    st.latex(ps["Fa_val"])

    st.subheader("Paso 4 · Restar y simplificar")
    st.latex(ps["resta"])
    st.latex(ps["resta_val"])
    st.latex(ps["final"])
    st.success(f"El volumen exacto del sólido de revolución es ${sp.latex(ps['V_sym'])}$ unidades cúbicas "
               f"(aproximadamente **{fmt(ps['V'], 4)}**).")
else:
    st.info("Esta integral no tiene antiderivada elemental: se usa SciPy (`quad`) como valor de referencia.")
    st.success(f"Volumen ≈ **{fmt(exacto, 8)}** unidades cúbicas.")

# =========================================================================== 4. COMPARACIÓN
st.header("4 · Comparación de métodos")
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
st.caption("Software realizado por Leyvi Revolorio, Marcos Arevalo, Derek Jimenez, Kevin Cax.")
