import numpy as np
import pandas as pd
import sympy as sp
from scipy.integrate import quad
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor,
)

x = sp.Symbol("x")
TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)
LOCAL = {
    "x": x, "pi": sp.pi, "e": sp.E, "E": sp.E,
    "sen": sp.sin, "tg": sp.tan, "ln": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
    "arcsin": sp.asin, "arccos": sp.acos, "arctan": sp.atan,
}

METODOS = ["Riemann izquierda", "Riemann derecha", "Punto medio", "Trapecio", "Simpson 1/3"]


# ===================================================================== entrada
def parsear_funcion(texto):
    """Convierte texto en expresión de SymPy (acepta ^, sen, ln, pi, 2x ...)."""
    expr = parse_expr(texto.replace("^", "**"), local_dict=LOCAL, transformations=TRANSFORMS)
    if not expr.free_symbols <= {x}:
        raise ValueError("Solo se permite la variable x.")
    return expr


def parsear_limite(texto):
    """Devuelve (valor simbólico, valor float). Acepta 2, -1.5, pi/2, sqrt(2)..."""
    sym = parse_expr(texto.strip(), local_dict=LOCAL, transformations=TRANSFORMS)
    if sym.free_symbols:
        raise ValueError("Los límites deben ser números (puedes usar pi, e, sqrt(2)...).")
    if sym.has(sp.Float):
        sym = sp.nsimplify(sym, rational=True)
    return sym, float(sp.N(sym))


def crear_f(expr):
    """Función numérica de NumPy. Resuelve singularidades removibles (sin(x)/x en 0)."""
    f = sp.lambdify(x, expr, modules=["numpy"])
    cache = {}

    def limite(p):
        if p not in cache:
            try:
                v = float(sp.limit(expr, x, p))
                cache[p] = v if np.isfinite(v) else np.nan
            except Exception:
                cache[p] = np.nan
        return cache[p]

    def g(t):
        t = np.asarray(t, dtype=float)
        with np.errstate(all="ignore"):
            y = np.asarray(f(t) * np.ones_like(t), dtype=float)
        malos = ~np.isfinite(y)
        if malos.any() and malos.sum() <= 5:
            for i in np.flatnonzero(malos):
                y.flat[i] = limite(float(t.flat[i]))
        return y
    return g


# ============================================================ formato de números
def fmt(v, dec=6):
    """Número -> texto corto para mostrar."""
    v = float(v)
    if np.isnan(v):
        return "nan"
    if np.isinf(v):
        return "∞" if v > 0 else "-∞"
    if v != 0 and abs(v) < 1e-4:
        return f"{v:.3e}".replace("e-0", "e-").replace("e+0", "e+")
    s = f"{v:.{dec}f}".rstrip("0").rstrip(".")
    return "0" if s in ("", "-0") else s


def _par(v):
    """Número entre paréntesis si es negativo (para sustituciones en LaTeX)."""
    s = fmt(v)
    if "e" in s:
        m, e = s.split("e")
        s = rf"{m}\times10^{{{int(e)}}}"
    return f"({s})" if float(v) < 0 else s


def _sl(sym):
    """Símbolo/valor a LaTeX, con paréntesis si es negativo."""
    s = sp.latex(sym)
    return f"({s})" if sym.is_negative else s


# ======================================================== integrales con _quad
def _quad(g, a, b, limit=200):
    def seguro(t):
        with np.errstate(all="ignore"):
            v = float(g(t))
        return v if np.isfinite(v) else 0.0
    val, _ = quad(seguro, a, b, limit=limit)
    return val


def area_total(f, a, b):
    """Área geométrica real (todas las regiones como positivas): ∫|f(x)| dx."""
    return _quad(lambda t: abs(f(t)), a, b, limit=300)


# ======================================================== solución exacta (TFC)
MSG_DIVERGE = ("La integral diverge o la función tiene una discontinuidad no integrable "
               "dentro de [a, b]. Elige un intervalo donde f sea continua.")


def verificar_divergencia(expr, a_sym, b_sym):
    """Lanza ValueError si hay asíntotas no integrables (1/x en [-1,1], tan(x) en [0,2])."""
    try:
        intervalo = sp.Interval(a_sym, b_sym)
        dom = sp.calculus.util.continuous_domain(expr, x, intervalo)
        singulares = intervalo - dom
        if not isinstance(singulares, sp.FiniteSet):
            return
        for p in singulares:
            li = sp.limit(expr, x, p, "-") if p > a_sym else 0
            ld = sp.limit(expr, x, p, "+") if p < b_sym else 0
            if not (li.is_finite and ld.is_finite):
                val = sp.integrate(expr, (x, a_sym, b_sym))
                if not val.is_finite:
                    raise ValueError(MSG_DIVERGE)
    except ValueError:
        raise
    except Exception:
        return


def _evaluar_F(F, p, lado):
    """F(p); si da indeterminado usa el límite lateral (integrales impropias)."""
    try:
        v = F.subs(x, p)
        if not v.has(sp.nan, sp.zoo, sp.oo, -sp.oo):
            return v
    except Exception:
        pass
    return sp.limit(F, x, p, lado)


def analisis_exacto(expr, f, a_sym, b_sym, a, b):
    """
    Devuelve dict con: anti, Fa, Fb, exacto_sym, exacto (float), fuente.
    Si no hay antiderivada elemental se usa scipy.quad como valor de referencia.
    """
    verificar_divergencia(expr, a_sym, b_sym)
    try:
        anti = sp.integrate(expr, x)
        if not anti.has(sp.Integral):
            Fb = sp.simplify(_evaluar_F(anti, b_sym, "-"))
            Fa = sp.simplify(_evaluar_F(anti, a_sym, "+"))
            ex_sym = sp.simplify(Fb - Fa)
            val = complex(sp.N(ex_sym))
            ref = _quad(f, a, b)
            if abs(val.imag) < 1e-12 and np.isfinite(val.real) \
                    and abs(val.real - ref) <= 1e-6 * max(1.0, abs(ref)):
                return dict(anti=anti, Fa=Fa, Fb=Fb, exacto_sym=ex_sym,
                            exacto=float(val.real), fuente="SymPy (Teorema Fundamental del Cálculo)")
    except Exception:
        pass
    val = _quad(f, a, b)
    if not np.isfinite(val):
        raise ValueError(MSG_DIVERGE)
    return dict(anti=None, Fa=None, Fb=None, exacto_sym=None, exacto=val,
                fuente="SciPy quad (no hay antiderivada elemental)")


# ====================================================== métodos numéricos
def nodos(nombre, a, b, n):
    """Nodos, coeficientes y factor de cada método:  resultado = factor · Σ coef·f(x)."""
    if nombre == "Simpson 1/3" and n % 2:
        n += 1  # Simpson requiere n par
    h = (b - a) / n
    if nombre == "Riemann izquierda":
        idx = np.arange(0, n); xs = a + h * idx; coef = np.ones(n); factor = h
    elif nombre == "Riemann derecha":
        idx = np.arange(1, n + 1); xs = a + h * idx; coef = np.ones(n); factor = h
    elif nombre == "Punto medio":
        idx = np.arange(0, n); xs = a + h * (idx + 0.5); coef = np.ones(n); factor = h
    elif nombre == "Trapecio":
        idx = np.arange(0, n + 1); xs = a + h * idx
        coef = np.full(n + 1, 2.0); coef[0] = coef[-1] = 1.0; factor = h / 2
    elif nombre == "Simpson 1/3":
        idx = np.arange(0, n + 1); xs = a + h * idx
        coef = np.ones(n + 1); coef[1:-1:2] = 4.0; coef[2:-1:2] = 2.0; factor = h / 3
    else:
        raise ValueError(f"Método desconocido: {nombre}")
    return dict(n=n, h=h, idx=idx, xs=xs, coef=coef, factor=factor)


def aproximar(f, a, b, n, nombre):
    """Solo el resultado numérico (rápido; se usa en la gráfica de convergencia)."""
    d = nodos(nombre, a, b, n)
    return float(d["factor"] * np.sum(d["coef"] * f(d["xs"])))


def calcular_metodo(f, a, b, n, nombre):
    """Resultado + todos los datos del procedimiento (tabla de nodos, suma, factor)."""
    d = nodos(nombre, a, b, n)
    ys = f(d["xs"])
    suma = float(np.sum(d["coef"] * ys))
    d.update(nombre=nombre, ys=ys, suma=suma, resultado=d["factor"] * suma)
    d["tabla"] = pd.DataFrame({
        "i": d["idx"],
        "x": d["xs"],
        "f(x)": ys,
        "Coef.": d["coef"].astype(int),
        "Coef · f(x)": d["coef"] * ys,
    })
    return d


# ========================================== procedimiento en LaTeX (paso a paso)
def _terminos(coef, ys, mostrar=4):
    items = [_par(y) if c == 1 else rf"{int(c)}\cdot{_par(y)}" for c, y in zip(coef, ys)]
    if len(items) > mostrar + 2:
        return " + ".join(items[:mostrar]) + r" + \cdots + " + " + ".join(items[-2:])
    return " + ".join(items)


def latex_metodo(m, a_sym, b_sym):
    """Devuelve las líneas LaTeX del procedimiento de un método numérico."""
    nombre, n, h = m["nombre"], m["n"], m["h"]
    A, B = _sl(a_sym), _sl(b_sym)
    delta = rf"\Delta x = \frac{{b-a}}{{n}} = \frac{{{B}-{A}}}{{{n}}} = {fmt(h)}"

    formulas = {
        "Riemann izquierda": r"S = \Delta x \sum_{i=0}^{n-1} f(x_i), \qquad x_i = a + i\,\Delta x",
        "Riemann derecha": r"S = \Delta x \sum_{i=1}^{n} f(x_i), \qquad x_i = a + i\,\Delta x",
        "Punto medio": r"S = \Delta x \sum_{i=0}^{n-1} f(\bar{x}_i), \qquad \bar{x}_i = a + \left(i+\tfrac{1}{2}\right)\Delta x",
        "Trapecio": r"S = \frac{\Delta x}{2}\left[ f(x_0) + 2\sum_{i=1}^{n-1} f(x_i) + f(x_n) \right]",
        "Simpson 1/3": r"S = \frac{\Delta x}{3}\left[ f(x_0) + 4\sum_{i\ \mathrm{impar}} f(x_i) + 2\sum_{i\ \mathrm{par}} f(x_i) + f(x_n) \right]",
    }
    fac_txt = {"Trapecio": rf"\frac{{{fmt(h)}}}{{2}}",
               "Simpson 1/3": rf"\frac{{{fmt(h)}}}{{3}}"}.get(nombre, fmt(h))
    sust = rf"S = {fac_txt}\left[ {_terminos(m['coef'], m['ys'])} \right]"
    suma = rf"\Sigma = \sum \text{{coef}}_i \, f(x_i) = {fmt(m['suma'])}"
    res = rf"S = {fmt(m['factor'])} \cdot {fmt(m['suma'])} = \mathbf{{{fmt(m['resultado'])}}}"
    return dict(delta=delta, formula=formulas[nombre], sustitucion=sust, suma=suma, resultado=res)


def latex_exacto(expr, ex, a_sym, b_sym):
    """Líneas LaTeX de la solución exacta por el Teorema Fundamental del Cálculo."""
    A, B = _sl(a_sym), _sl(b_sym)
    Fb = float(sp.N(ex["Fb"])) if ex["Fb"].is_real else complex(sp.N(ex["Fb"])).real
    Fa = float(sp.N(ex["Fa"])) if ex["Fa"].is_real else complex(sp.N(ex["Fa"])).real
    return dict(
        antiderivada=rf"F(x) = \int {sp.latex(expr)}\,dx = {sp.latex(ex['anti'])} + C",
        tfc=r"\int_a^b f(x)\,dx = F(b) - F(a)",
        Fb=rf"F({B}) = {sp.latex(ex['Fb'])} \approx {fmt(Fb)}",
        Fa=rf"F({A}) = {sp.latex(ex['Fa'])} \approx {fmt(Fa)}",
        resultado=rf"\int_{{{sp.latex(a_sym)}}}^{{{sp.latex(b_sym)}}} f(x)\,dx = F({B}) - F({A}) "
                  rf"= {sp.latex(ex['exacto_sym'])} \approx \mathbf{{{fmt(ex['exacto'])}}}",
    )
