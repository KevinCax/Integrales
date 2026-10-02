import numpy as np
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


if __name__ == "__main__":
    for nombre, fn in list(globals().items()):
        if nombre.startswith("test_"):
            fn()
            print("OK ", nombre)
    print("Todas las pruebas pasaron.")