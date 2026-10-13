import numpy as np
import sympy as sp
from calculo import *


def volumen(texto, a_txt, b_txt, k_txt="0", g_txt=None):
    expr = parsear_funcion(texto)
    a_sym, a = parsear_limite(a_txt)
    b_sym, b = parsear_limite(b_txt)
    k_sym, k = parsear_limite(k_txt)
    g_expr = parsear_funcion(g_txt) if g_txt else k_sym
    prob = preparar_problema(expr, g_expr, k_sym, a, b)
    return prob["integrando"], crear_f(prob["integrando"]), a_sym, b_sym, a, b


def test_volumenes_conocidos():
    casos = [("x", "0", "3", "0", 9 * np.pi),                 # cono
             ("2", "0", "3", "0", 12 * np.pi),                # cilindro
             ("sqrt(1-x^2)", "-1", "1", "0", 4 * np.pi / 3),  # esfera
             ("sqrt(x)", "0", "4", "0", 8 * np.pi),           # paraboloide
             ("x^2", "0", "1", "1", 8 * np.pi / 15),          # giro alrededor de y = 1
             ("exp(-x)", "0", "1", "0", np.pi * (1 - np.exp(-2)) / 2)]
    for t, a, b, k, esperado in casos:
        integ, g, a_s, b_s, a_f, b_f = volumen(t, a, b, k)
        ex = analisis_exacto(integ, g, a_s, b_s, a_f, b_f)
        assert abs(ex["exacto"] - esperado) < 1e-6, (t, ex["exacto"], esperado)


def test_arandelas():
    # ejemplo: R = x, r = 1/x en [1, 4]  ->  V = 81π/4
    integ, g, a_s, b_s, a, b = volumen("x", "1", "4", "0", "1/x")
    ex = analisis_exacto(integ, g, a_s, b_s, a, b)
    assert abs(ex["exacto"] - 81 * np.pi / 4) < 1e-9
    # la función exterior se detecta sola aunque se escriba al revés
    prob = preparar_problema(parsear_funcion("1/x"), parsear_funcion("x"), sp.Integer(0), 1, 4)
    assert prob["intercambio"] and not prob["disco"]
    # pasos a mano
    p = preparar_problema(parsear_funcion("x"), parsear_funcion("1/x"), sp.Integer(0), 1, 4)
    ps = pasos_solucion(p, sp.Integer(0), sp.Integer(1), sp.Integer(4))
    assert ps["V_sym"] == 81 * sp.pi / 4
    # eje desplazado: f = x, g = 0, giro en y = -2 en [0, 3] -> arandela, V = 27π
    integ, g, a_s, b_s, a, b = volumen("x", "0", "3", "-2", "0")
    assert abs(analisis_exacto(integ, g, a_s, b_s, a, b)["exacto"] - 27 * np.pi) < 1e-9


def test_metodos_convergen():
    integ, g, a_s, b_s, a, b = volumen("x", "0", "3")          # cono: V = 9π
    V = 9 * np.pi
    assert abs(aproximar(g, a, b, 100, "Simpson 1/3") - V) < 1e-9   # integrando grado 2: exacto
    for m in METODOS:
        assert abs(aproximar(g, a, b, 4000, m) - V) < 0.05, m
    assert aproximar(g, a, b, 50, "Discos (izquierda)") < V < aproximar(g, a, b, 50, "Discos (derecha)")


def test_simpson_n_impar():
    assert nodos("Simpson 1/3", 0, 3, 7)["n"] == 8


def test_divergentes():
    for t, a, b in [("1/x", "-1", "1"), ("tan(x)", "0", "2"), ("1/(x-1)", "0", "2")]:
        integ, g, a_s, b_s, a_f, b_f = volumen(t, a, b)
        try:
            analisis_exacto(integ, g, a_s, b_s, a_f, b_f)
        except ValueError:
            continue
        raise AssertionError(f"{t} debía detectarse como divergente")


def test_entrada_invalida():
    for t in ["y + 1", "x + z"]:
        try:
            parsear_funcion(t)
        except ValueError:
            continue
        raise AssertionError(t)


def volumen_vertical(texto, lo, hi, k="0", g_txt=None):
    """Giro alrededor de x = k: se despeja x = g(y) y se integra en y."""
    f = invertir(parsear_funcion(texto), float(sp.sympify(lo)), float(sp.sympify(hi)))["expr"]
    g = invertir(parsear_funcion(g_txt), float(sp.sympify(lo)), float(sp.sympify(hi)))["expr"] if g_txt else None
    k_sym, _ = parsear_limite(k)
    a_sym, a = parsear_limite(lo)
    b_sym, b = parsear_limite(hi)
    prob = preparar_problema(f, g if g is not None else k_sym, k_sym, a, b)
    return analisis_exacto(prob["integrando"], crear_f(prob["integrando"]), a_sym, b_sym, a, b)["exacto"]


def test_eje_vertical():
    assert abs(volumen_vertical("x^2", "0", "4") - 8 * np.pi) < 1e-9            # paraboloide
    assert abs(volumen_vertical("x^3", "0", "8") - 19.2 * np.pi) < 1e-9
    assert abs(volumen_vertical("x^2", "0", "4", "-1") - 68 * np.pi / 3) < 1e-9  # eje x = -1
    assert abs(volumen_vertical("sqrt(x)", "0", "2") - 32 * np.pi / 5) < 1e-9
    assert abs(volumen_vertical("x^2", "0", "1", "0", "sqrt(x)") - 0.3 * np.pi) < 1e-9   # arandelas


def test_inversion():
    assert sp.simplify(invertir(parsear_funcion("x^2"), 0, 4)["expr"] - sp.sqrt(x)) == 0
    assert sp.simplify(invertir(parsear_funcion("2x+1"), 0, 5)["expr"] - (x - 1) / 2) == 0
    try:
        invertir(parsear_funcion("x^2+x^4"), -1, 0.5)      # no es uno a uno en ese rango
    except ValueError:
        return
    raise AssertionError("debía rechazar una función no invertible")


def test_sintaxis_flexible():
    for t, esperado in [("e^x", sp.exp(x)), ("sen(x)", sp.sin(x)), ("1,5x", 1.5 * x), ("x²", x ** 2),
                        ("2xsin(x)", 2 * x * sp.sin(x)), ("√(x)", sp.sqrt(x)), ("raiz(4-x^2)", sp.sqrt(4 - x ** 2))]:
        assert sp.simplify(parsear_funcion(t) - esperado) == 0, t


def test_entrada_peligrosa():
    for t in ['__import__("os").system("dir")', "x.__class__", "open('a')", "foo(x)", "sin", "(x", "x+"]:
        try:
            parsear_funcion(t)
        except ValueError:
            continue
        raise AssertionError(f"{t!r} debía rechazarse")


def volumen_region(texto, vertical=False, k="0"):
    k_sym, _ = parsear_limite(k)
    r = parsear_region(texto, vertical, k_sym)
    prob = preparar_problema(r["f"], r["g"], k_sym, r["a"], r["b"])
    ex = analisis_exacto(prob["integrando"], crear_f(prob["integrando"]), r["a_sym"], r["b_sym"], r["a"], r["b"])
    return ex["exacto"], r


def test_region_por_fronteras():
    v, r = volumen_region("y=(x-2)^2; x=0; y=0")          # ejercicio del enunciado
    assert (r["a_sym"], r["b_sym"]) == (0, 2) and abs(v - 32 * np.pi / 5) < 1e-9
    assert abs(volumen_region("y=x; x=3; y=0")[0] - 9 * np.pi) < 1e-9
    assert abs(volumen_region("y=x; y=x^2")[0] - 2 * np.pi / 15) < 1e-9              # límites = puntos de corte
    assert abs(volumen_region("y=x; y=1/x; x=1; x=4")[0] - 81 * np.pi / 4) < 1e-9
    assert abs(volumen_region("y=sin(x); y=0; x=0; x=pi")[0] - np.pi ** 2 / 2) < 1e-9
    assert abs(volumen_region("y=x^2; x=0; x=1; y=0", k="2")[0] - 17 * np.pi / 15) < 1e-9
    assert abs(volumen_region("y=x^2; x=0; y=4", vertical=True)[0] - 8 * np.pi) < 1e-9
    assert abs(volumen_region("y=(x-2)^2; x=0; y=0", vertical=True)[0] - 8 * np.pi / 3) < 1e-9


def test_region_invalida():
    for t in ["y=x^2", "y=x^2; z=1", "y=x^2 = 4", "x=1"]:
        try:
            parsear_region(t, False, sp.Integer(0))
        except ValueError:
            continue
        raise AssertionError(f"{t!r} debía rechazarse")


def resolver_enunciado(texto):
    """Como la app: toma el eje del enunciado, halla límites y calcula el volumen exacto."""
    eje = extraer_eje(texto)
    vertical, k_txt = eje if eje else (False, "0")
    k_sym, _ = parsear_limite(k_txt)
    r = parsear_region(texto, vertical, k_sym)
    prob = preparar_problema(r["f"], r["g"], k_sym, r["a"], r["b"])
    return analisis_exacto(prob["integrando"], crear_f(prob["integrando"]),
                           r["a_sym"], r["b_sym"], r["a"], r["b"])["exacto"], r


def test_enunciados_completos():
    pi = np.pi
    casos = [("y=x^3-x, y=0, eje x", 16 * pi / 105, (-1, 1)),                 # tres cortes: -1, 0, 1
             ("y=e^(-x), x=1, y=1, eje y=2", pi * (4 / np.e - 0.5 - np.e ** -2 / 2), (0, 1)),
             ("y=|cos(x)|, y=0, 0<=x<=2pi, eje x", pi ** 2, (0, 2 * pi)),
             ("y=abs(cos x), y=0, 0 ≤ x ≤ 2π, eje x", pi ** 2, (0, 2 * pi)),
             ("y=tan(x), y=0, x=pi/4; eje x", pi * (1 - pi / 4), (0, pi / 4)),
             ("y=tan x, y=0, x=pi/4, eje x", pi * (1 - pi / 4), (0, pi / 4)),
             ("y=x^2, x=0, y=4, eje y", 8 * pi, (0, 4)),
             ("y=x^2\nx=0\ny=4\neje y", 8 * pi, (0, 4))]
    for texto, esperado, (a, b) in casos:
        v, r = resolver_enunciado(texto)
        assert abs(v - esperado) < 1e-9, (texto, v, esperado)
        assert abs(r["a"] - a) < 1e-9 and abs(r["b"] - b) < 1e-9, (texto, r["a"], r["b"])


def test_eje_en_el_enunciado():
    assert extraer_eje("y=x^2, y=0, eje x") == (False, "0")
    assert extraer_eje("y=x^2, x=0, eje y") == (True, "0")
    assert extraer_eje("y=x^2, x=1, eje y=2") == (False, "2")
    assert extraer_eje("y=x^2, y=1, eje x=-1") == (True, "-1")
    assert extraer_eje("y=x^2, x=0") is None


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK ", nombre)
    print("Todas las pruebas pasaron.")