import numpy as np
from calculo import *


def integral(texto, a_txt, b_txt):
    expr = parsear_funcion(texto)
    a_sym, a = parsear_limite(a_txt)
    b_sym, b = parsear_limite(b_txt)
    f = crear_f(expr)
    return expr, f, a_sym, b_sym, a, b


def test_exactos():
    casos = [("x^2", "0", "3", 9.0), ("sin(x)", "0", "pi", 2.0), ("2x+1", "-1", "2", 6.0),
             ("exp(x)", "0", "1", np.e - 1), ("1/x", "1", "e", 1.0), ("sin(x)/x", "0", "pi", 1.851937052)]
    for t, a, b, esperado in casos:
        expr, f, a_s, b_s, a_f, b_f = integral(t, a, b)
        ex = analisis_exacto(expr, f, a_s, b_s, a_f, b_f)
        assert abs(ex["exacto"] - esperado) < 1e-6, (t, ex["exacto"], esperado)


def test_metodos_convergen():
    expr, f, a_s, b_s, a, b = integral("x^2", "0", "3")
    assert abs(aproximar(f, a, b, 100, "Simpson 1/3") - 9.0) < 1e-10      # exacto para polinomios grado <= 3
    for m in METODOS:
        assert abs(aproximar(f, a, b, 2000, m) - 9.0) < 0.01, m
    assert aproximar(f, a, b, 100, "Riemann izquierda") < 9 < aproximar(f, a, b, 100, "Riemann derecha")


def test_simpson_n_impar():
    expr, f, *_ = integral("x^2", "0", "3")
    assert nodos("Simpson 1/3", 0, 3, 7)["n"] == 8


def test_divergentes():
    for t, a, b in [("1/x", "-1", "1"), ("tan(x)", "0", "2"), ("1/(x-1)", "0", "2")]:
        expr, f, a_s, b_s, a_f, b_f = integral(t, a, b)
        try:
            analisis_exacto(expr, f, a_s, b_s, a_f, b_f)
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
