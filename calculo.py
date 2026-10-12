import re

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

METODOS = ["Discos (izquierda)", "Discos (derecha)", "Discos (punto medio)", "Trapecio", "Simpson 1/3"]


# ===================================================================== entrada
NOMBRES_OK = set(LOCAL) | {"sin", "cos", "tan", "cot", "sec", "csc", "exp", "log",
                           "sinh", "cosh", "tanh", "asin", "acos", "atan"}
FUNCIONES = NOMBRES_OK - {"x", "pi", "e", "E"}


def _partir(palabra):
    """'xsin' -> ['x', 'sin'];  'pix' -> ['pi', 'x']. None si no se puede."""
    mejor = [None] * (len(palabra) + 1)
    mejor[0] = []
    for i in range(1, len(palabra) + 1):
        for j in range(i):
            tr = palabra[j:i]
            if mejor[j] is not None and tr in NOMBRES_OK:
                mejor[i] = mejor[j] + [tr]
                break
    return mejor[-1]


def normalizar(texto):
    """Limpia el texto y revisa que solo tenga funciones/constantes conocidas (evita código malicioso)."""
    t = texto.strip().lower()
    if not t:
        raise ValueError("Escribe una función (por ejemplo: x^2).")
    if len(t) > 200:
        raise ValueError("La expresión es demasiado larga.")
    for a, b in (("²", "^2"), ("³", "^3"), ("π", "pi"), ("√", "sqrt"), ("raíz", "sqrt"), ("raiz", "sqrt"),
                 ("−", "-"), ("×", "*"), ("÷", "/"), ("·", "*"), ("[", "("), ("]", ")"), ("{", "("), ("}", ")")):
        t = t.replace(a, b)
    t = re.sub(r"(?<=\d),(?=\d)", ".", t)                     # 1,5 -> 1.5
    if t.count("(") != t.count(")"):
        raise ValueError("Los paréntesis no están balanceados: revisa que cada ( tenga su ).")
    if not re.fullmatch(r"[a-z0-9_+\-*/^().\s]+", t):
        malo = sorted(set(re.sub(r"[a-z0-9_+\-*/^().\s]", "", t)))
        raise ValueError(f"Carácter no permitido: {' '.join(malo)}")

    def arreglar(m):
        palabra, resto = m.group(0), t[m.end():].lstrip()
        trozos = _partir(palabra)
        if trozos is None:
            raise ValueError(f"No reconozco «{palabra}». Usa: x, pi, e, sin, cos, tan, exp, ln, sqrt, abs...")
        for i, tr in enumerate(trozos):
            if tr in FUNCIONES and (i < len(trozos) - 1 or not resto.startswith("(")):
                raise ValueError(f"Escribe la función con paréntesis, por ejemplo {tr}(x).")
        return "*".join(trozos)
    return re.sub(r"[a-z_]+", arreglar, t)


def parsear_funcion(texto):
    """Convierte texto en expresión de SymPy (acepta ^, sen, ln, pi, 2x, 1,5, e^x ...)."""
    try:
        expr = parse_expr(normalizar(texto), local_dict=LOCAL, transformations=TRANSFORMS)
    except (SyntaxError, TypeError, sp.SympifyError, AttributeError, NameError) as err:
        raise ValueError("No pude entender la expresión. Revisa los operadores y paréntesis "
                         "(ejemplo: 2*x^2 + sin(x)).") from err
    if not expr.free_symbols <= {x}:
        raise ValueError("Solo se permite la variable x.")
    return expr


def parsear_limite(texto):
    """Devuelve (valor simbólico, valor float). Acepta 2, -1.5, pi/2, sqrt(2)..."""
    try:
        sym = parse_expr(normalizar(texto), local_dict=LOCAL, transformations=TRANSFORMS)
    except (SyntaxError, TypeError, sp.SympifyError, AttributeError, NameError) as err:
        raise ValueError(f"No pude entender «{texto}». Escribe un número (2, -1.5, pi/2, sqrt(2)...).") from err
    if sym.free_symbols:
        raise ValueError("Los límites deben ser números (puedes usar pi, e, sqrt(2)...).")
    if sym.has(sp.Float):
        sym = sp.nsimplify(sym, rational=True)
    return sym, float(sp.N(sym))


def invertir(expr, lo, hi):
    """
    Despeja x de y = f(x) para girar alrededor de un eje vertical (el método de discos integra en y).
    Devuelve dict(expr=x(y) en la variable interna x, latex=x(y) en LaTeX con y, ramas=n).
    Lanza ValueError si f no se puede invertir en [lo, hi].
    """
    yy = sp.Symbol("yy")
    try:
        sols = sp.solve(sp.Eq(yy, expr), x)
    except Exception:
        sols = []
    f = sp.lambdify(x, expr, modules=["numpy"])
    ys = np.linspace(lo, hi, 41)[1:-1]
    buenas = []
    for s in sols:
        try:
            h = sp.lambdify(yy, s, modules=["numpy"])
            with np.errstate(all="ignore"):
                z = np.asarray(h(ys.astype(complex)) * np.ones_like(ys, dtype=complex))
                if np.max(np.abs(z.imag)) > 1e-9 or not np.all(np.isfinite(z)):
                    continue
                xr = z.real
                if np.max(np.abs(f(xr) - ys)) > 1e-6 * (1 + np.max(np.abs(ys))):
                    continue
        except Exception:
            continue
        buenas.append((s, xr))
    if not buenas:
        raise ValueError("No pude despejar x en función de y para esa curva en ese intervalo de y "
                         "(la función puede no ser invertible ahí). Prueba con otro intervalo o con "
                         "una curva que sea creciente o decreciente.")
    buenas.sort(key=lambda t: (float(np.mean(t[1])) < 0, float(np.mean(np.abs(t[1])))))   # rama positiva y la más cercana al origen
    s = buenas[0][0]
    return dict(expr=s.subs(yy, x), latex=sp.latex(s.subs(yy, sp.Symbol("y"))), ramas=len(buenas))


def a_var(texto, v="x"):
    """Cambia la variable x por v en un texto LaTeX (para mostrar y en vez de x en giros verticales)."""
    if v == "x":
        return texto
    texto = re.sub(r"(?<=[\s,])dx(?![A-Za-z])", "d" + v, texto)
    return re.sub(r"(?<![A-Za-z\\])x(?![A-Za-z])", v, texto)


def preparar_problema(f_expr, g_expr, k_sym, a, b):
    """
    Dadas dos curvas f y g y el eje y = k, decide cuál es la exterior (R) y cuál la
    interior (r) según su distancia al eje, y arma el integrando A(x) = π([R-k]² - [r-k]²).
    """
    xs = np.linspace(a, b, 201)[1:-1]
    ff = crear_f(f_expr - k_sym)(xs)
    gg = crear_f(g_expr - k_sym)(xs)
    f_afuera = np.nanmean(ff ** 2) >= np.nanmean(gg ** 2)
    R, r = (f_expr, g_expr) if f_afuera else (g_expr, f_expr)
    Rk, rk = R - k_sym, r - k_sym
    h = sp.expand(Rk ** 2 - rk ** 2)           # integrando sin el factor π
    return dict(R=R, r=r, Rk=Rk, rk=rk, h=h, integrando=sp.pi * h,
                intercambio=not f_afuera, disco=bool(sp.simplify(rk) == 0))


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
    if nombre == "Discos (izquierda)":
        idx = np.arange(0, n); xs = a + h * idx; coef = np.ones(n); factor = h
    elif nombre == "Discos (derecha)":
        idx = np.arange(1, n + 1); xs = a + h * idx; coef = np.ones(n); factor = h
    elif nombre == "Discos (punto medio)":
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


def calcular_metodo(g, a, b, n, nombre, columnas=None):
    """
    g(x) = A(x) = π([R-k]² - [r-k]²).  Devuelve resultado + todos los datos del procedimiento
    (tabla de nodos, suma, factor). `columnas` = {"R(x)": función, "r(x)": función} (solo para la tabla).
    """
    d = nodos(nombre, a, b, n)
    ys = g(d["xs"])
    suma = float(np.sum(d["coef"] * ys))
    d.update(nombre=nombre, ys=ys, suma=suma, resultado=d["factor"] * suma)
    cols = {"i": d["idx"], "x": d["xs"]}
    for nom, fun in (columnas or {}).items():
        cols[nom] = np.abs(fun(d["xs"]))
    cols["A(x)"] = ys
    cols["Coef."] = d["coef"].astype(int)
    cols["Coef · A(x)"] = d["coef"] * ys
    d["tabla"] = pd.DataFrame(cols)
    return d


# ========================================== procedimiento en LaTeX (paso a paso)
def _terminos(coef, ys, mostrar=4):
    items = [_par(y) if c == 1 else rf"{int(c)}\cdot{_par(y)}" for c, y in zip(coef, ys)]
    if len(items) > mostrar + 2:
        return " + ".join(items[:mostrar]) + r" + \cdots + " + " + ".join(items[-2:])
    return " + ".join(items)


def latex_metodo(m, a_sym, b_sym, v="x"):
    """Devuelve las líneas LaTeX del procedimiento de un método numérico."""
    nombre, n, h = m["nombre"], m["n"], m["h"]
    A, B = _sl(a_sym), _sl(b_sym)
    delta = rf"\Delta x = \frac{{b-a}}{{n}} = \frac{{{B}-{A}}}{{{n}}} = {fmt(h)}"

    area = r"A(x) = \pi\left([R(x)]^2 - [r(x)]^2\right)"
    formulas = {
        "Discos (izquierda)": r"V \approx \Delta x \sum_{i=0}^{n-1} A(x_i), \qquad x_i = a + i\,\Delta x",
        "Discos (derecha)": r"V \approx \Delta x \sum_{i=1}^{n} A(x_i), \qquad x_i = a + i\,\Delta x",
        "Discos (punto medio)": r"V \approx \Delta x \sum_{i=0}^{n-1} A(\bar{x}_i), \qquad \bar{x}_i = a + \left(i+\tfrac{1}{2}\right)\Delta x",
        "Trapecio": r"V \approx \frac{\Delta x}{2}\left[ A(x_0) + 2\sum_{i=1}^{n-1} A(x_i) + A(x_n) \right]",
        "Simpson 1/3": r"V \approx \frac{\Delta x}{3}\left[ A(x_0) + 4\sum_{i\ \mathrm{impar}} A(x_i) + 2\sum_{i\ \mathrm{par}} A(x_i) + A(x_n) \right]",
    }
    fac_txt = {"Trapecio": rf"\frac{{{fmt(h)}}}{{2}}",
               "Simpson 1/3": rf"\frac{{{fmt(h)}}}{{3}}"}.get(nombre, fmt(h))
    sust = rf"V \approx {fac_txt}\left[ {_terminos(m['coef'], m['ys'])} \right]"
    suma = rf"\Sigma = \sum \text{{coef}}_i \, A(x_i) = {fmt(m['suma'])}"
    res = rf"V \approx {fmt(m['factor'])} \cdot {fmt(m['suma'])} = \mathbf{{{fmt(m['resultado'])}}}\ \text{{u}}^3"
    L = dict(area=area, delta=delta, formula=formulas[nombre], sustitucion=sust, suma=suma, resultado=res)
    return {k: a_var(t, v) for k, t in L.items()}


# ============================================ solución exacta paso a paso (como a mano)
def _sust(F, p):
    """F evaluada en p SIN simplificar (para mostrar '4^3/3 + 1/4' y luego el valor)."""
    s = sp.latex(p)
    if not (p.is_Integer and p >= 0):
        s = rf"\left({s}\right)"
    return F.subs(x, sp.Symbol(s))


def pasos_solucion(prob, k_sym, a_sym, b_sym, v="x"):
    """
    Resuelve V = π ∫ (R² - r²) dx como se haría a mano. Devuelve un dict con las líneas LaTeX
    o None si SymPy no encuentra una antiderivada elemental.
    """
    h = prob["h"]
    try:
        partes = [(t, sp.integrate(t, x)) for t in sp.Add.make_args(h)]
        if any(F.has(sp.Integral) for _, F in partes):
            return None
        F = sp.Add(*[F for _, F in partes])
        Fb = sp.simplify(_evaluar_F(F, b_sym, "-"))
        Fa = sp.simplify(_evaluar_F(F, a_sym, "+"))
        dif = sp.simplify(Fb - Fa)
        V = sp.simplify(sp.pi * dif)
        valor = complex(sp.N(V))
        if abs(valor.imag) > 1e-12 or not np.isfinite(valor.real):
            return None
    except Exception:
        return None

    A, B = _sl(a_sym), _sl(b_sym)
    la, lb = sp.latex(a_sym), sp.latex(b_sym)
    Rk, rk = prob["Rk"], prob["rk"]
    if prob["disco"]:
        plantea = rf"V = \pi \int_{{{la}}}^{{{lb}}} \left[{sp.latex(Rk)}\right]^2 dx"
    else:
        plantea = (rf"V = \pi \int_{{{la}}}^{{{lb}}} \left( \left[{sp.latex(Rk)}\right]^2 "
                   rf"- \left[{sp.latex(rk)}\right]^2 \right) dx")
    general = (r"V = \pi \int_a^b [R(x)]^2\,dx" if prob["disco"]
               else r"V = \pi \int_a^b \left( [R(x)]^2 - [r(x)]^2 \right) dx")
    P = dict(
        general=general,
        plantea=plantea,
        simplificada=rf"V = \pi \int_{{{la}}}^{{{lb}}} \left({sp.latex(h)}\right) dx",
        terminos=[(sp.latex(t), sp.latex(Fi)) for t, Fi in partes],
        corchete=rf"V = \pi \left[ {sp.latex(F)} \right]_{{{la}}}^{{{lb}}}",
        Fb_sust=rf"F({B}) = {sp.latex(_sust(F, b_sym))}",
        Fb_val=rf"F({B}) = {sp.latex(Fb)}",
        Fa_sust=rf"F({A}) = {sp.latex(_sust(F, a_sym))}",
        Fa_val=rf"F({A}) = {sp.latex(Fa)}",
        resta=rf"V = \pi\left( {sp.latex(Fb)} - {_par_sym(Fa)} \right)",
        resta_val=rf"V = \pi \left( {sp.latex(dif)} \right)",
        final=rf"V = {sp.latex(V)}",
        V_sym=V, V=float(valor.real),
    )
    cadenas = {k: a_var(t, v) for k, t in P.items() if isinstance(t, str)}
    cadenas["terminos"] = [(a_var(t, v), a_var(Fi, v)) for t, Fi in P["terminos"]]
    return {**P, **cadenas}


def _par_sym(e):
    s = sp.latex(e)
    return rf"\left({s}\right)" if e.is_negative else s