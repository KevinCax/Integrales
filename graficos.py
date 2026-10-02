import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calculo import METODOS, aproximar, fmt

AZUL, NARANJA, ROJO, VERDE = "#1f77b4", "#ff9f43", "#d62728", "#2ca02c"
METODOS_DISCO = ("Discos (izquierda)", "Discos (derecha)", "Discos (punto medio)")


def _lim(v):
    """Reemplaza valores no finitos por NaN (Plotly los deja como huecos)."""
    v = np.asarray(v, dtype=float)
    return np.where(np.isfinite(v), v, np.nan)


# =============================================================== PLOTLY 2D
def fig_region_2d(Rk, rk, a, b, k, m=None, nombres=("R(x)", "r(x)")):
    """
    Región entre las dos curvas, eje de giro y reflejo (silueta del sólido).
    Rk, rk = radios con signo (curva - k). Si `m` es un método de discos, dibuja
    las secciones (arandelas vistas de lado) con n <= 100.
    """
    X = np.linspace(a, b, 600)
    yR, yr = _lim(k + Rk(X)), _lim(k + rk(X))
    fig = go.Figure()
    # región original
    fig.add_trace(go.Scatter(x=X, y=yr, mode="lines", line=dict(color=ROJO, width=2), name=nombres[1]))
    fig.add_trace(go.Scatter(x=X, y=yR, mode="lines", line=dict(color=AZUL, width=3), name=nombres[0],
                             fill="tonexty", fillcolor="rgba(31,119,180,0.30)"))
    # reflejo respecto al eje
    fig.add_trace(go.Scatter(x=X, y=2 * k - yr, mode="lines", line=dict(color=ROJO, width=1, dash="dot"),
                             name="reflejo r", showlegend=False))
    fig.add_trace(go.Scatter(x=X, y=2 * k - yR, mode="lines", line=dict(color=AZUL, width=1, dash="dot"),
                             name="reflejo R", showlegend=False,
                             fill="tonexty", fillcolor="rgba(31,119,180,0.10)"))
    # secciones del método
    if m is not None and m["nombre"] in METODOS_DISCO and m["n"] <= 100:
        h = m["h"]
        for i in range(m["n"]):
            xe = np.array([m["xs"][i]])
            ro, ri = abs(float(Rk(xe)[0])), abs(float(rk(xe)[0]))
            if not (np.isfinite(ro) and np.isfinite(ri)):
                continue
            for s in (1, -1):
                y0, y1 = sorted([k + s * ri, k + s * ro])
                fig.add_shape(type="rect", x0=a + i * h, x1=a + (i + 1) * h, y0=y0, y1=y1,
                              line=dict(color=ROJO, width=1), fillcolor="rgba(255,159,67,0.45)")
    fig.add_hline(y=k, line=dict(color=ROJO, dash="dash", width=1.5),
                  annotation_text=f"eje de giro y = {fmt(k)}", annotation_position="top left")
    fig.add_vline(x=a, line=dict(color="gray", dash="dash", width=1))
    fig.add_vline(x=b, line=dict(color="gray", dash="dash", width=1))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      xaxis_title="x", yaxis_title="y", legend=dict(orientation="h", y=1.1),
                      title=dict(text="Región (2D)", x=0.02, font=dict(size=14)))
    return fig


# =============================================================== PLOTLY 3D
def _surf(XX, rho, TT, k, color, opacity):
    return go.Surface(x=XX, y=k + rho * np.cos(TT), z=rho * np.sin(TT),
                      colorscale=[[0, color], [1, color]], showscale=False, opacity=opacity,
                      hoverinfo="skip", lighting=dict(ambient=0.6, diffuse=0.8, specular=0.2))


def _lado(x0, x1, rho, th, k, color, opacity):
    """Superficie cilíndrica entre x0 y x1 con radio constante."""
    T = len(th)
    return _surf(np.vstack([np.full(T, x0), np.full(T, x1)]), np.full((2, T), rho),
                 np.vstack([th, th]), k, color, opacity)


def _tapa(x0, ri, ro, th, k, color, opacity):
    """Corona circular (arandela) en x = x0 entre radios ri y ro."""
    T = len(th)
    return _surf(np.full((2, T), x0), np.vstack([np.full(T, ri), np.full(T, ro)]),
                 np.vstack([th, th]), k, color, opacity)


def fig_solido_3d(Rk, rk, a, b, k, corte=False, m=None, titulo="Sólido de revolución"):
    """
    Sólido 3D interactivo (se puede girar, acercar y mover con el mouse).
    - corte=True: quita 1/4 del sólido para ver el interior (y la arandela).
    - m = método de discos con n <= 30: dibuja los discos/arandelas apilados.
    """
    X = np.linspace(a, b, 160)
    R, r = np.abs(_lim(Rk(X))), np.abs(_lim(rk(X)))
    R, r = np.nan_to_num(R), np.nan_to_num(r)
    hueco = bool(np.max(r) > 1e-12)
    th = np.linspace(0, 1.5 * np.pi if corte else 2 * np.pi, 70)
    fig = go.Figure()

    if m is not None and m["nombre"] in METODOS_DISCO and m["n"] <= 30:
        h = m["h"]
        th = np.linspace(0, 1.5 * np.pi if corte else 2 * np.pi, 48)
        for i in range(m["n"]):
            xl, xr = a + i * h, a + (i + 1) * h
            xe = np.array([m["xs"][i]])
            ro, ri = abs(float(Rk(xe)[0])), abs(float(rk(xe)[0]))
            if not (np.isfinite(ro) and np.isfinite(ri)):
                continue
            fig.add_trace(_lado(xl, xr, ro, th, k, NARANJA, 0.9))
            if ri > 1e-12:
                fig.add_trace(_lado(xl, xr, ri, th, k, "#d9822b", 0.9))
            fig.add_trace(_tapa(xl, ri, ro, th, k, "#ffb870", 1.0))
            fig.add_trace(_tapa(xr, ri, ro, th, k, "#ffb870", 1.0))
    else:
        Xg, Tg = np.meshgrid(X, th)
        fig.add_trace(_surf(Xg, np.tile(R, (len(th), 1)), Tg, k, AZUL, 0.55))                 # exterior
        if hueco:
            fig.add_trace(_surf(Xg, np.tile(r, (len(th), 1)), Tg, k, "#e8743b", 0.75))         # interior
        for x0 in (a, b):                                                                      # tapas
            fig.add_trace(_tapa(x0, float(np.interp(x0, X, r)), float(np.interp(x0, X, R)), th, k, "#7fb8e6", 0.9))
        if corte:                                                                              # caras del corte
            for ang in (0.0, th[-1]):
                rho = np.vstack([r, R])
                XX = np.vstack([X, X])
                fig.add_trace(go.Surface(x=XX, y=k + rho * np.cos(ang), z=rho * np.sin(ang),
                                         colorscale=[[0, "#dddddd"], [1, "#dddddd"]], showscale=False,
                                         opacity=1.0, hoverinfo="skip"))

    # curvas generadoras (plano z = 0) y eje de giro
    fig.add_trace(go.Scatter3d(x=X, y=k + np.nan_to_num(_lim(Rk(X))), z=np.zeros_like(X), mode="lines",
                               line=dict(color=AZUL, width=6), name="R(x)"))
    if hueco:
        fig.add_trace(go.Scatter3d(x=X, y=k + np.nan_to_num(_lim(rk(X))), z=np.zeros_like(X), mode="lines",
                                   line=dict(color=ROJO, width=6), name="r(x)"))
    fig.add_trace(go.Scatter3d(x=[a, b], y=[k, k], z=[0, 0], mode="lines",
                               line=dict(color="black", width=4, dash="dash"), name="eje de giro"))
    fig.update_layout(
        height=500, margin=dict(l=0, r=0, t=35, b=0), showlegend=False,
        title=dict(text=titulo, x=0.02, font=dict(size=14)),
        scene=dict(aspectmode="data", xaxis_title="x", yaxis_title="y", zaxis_title="z",
                   camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))))
    return fig


# =========================================================== MATPLOTLIB (2D)
def _curva(f, a, b):
    xs = np.linspace(a, b, 1200)
    ys = f(xs)
    return xs, np.where(np.isfinite(ys), ys, np.nan)


def _ejes(ax, ys, extra=()):
    vals = np.concatenate([ys[np.isfinite(ys)], [0.0], np.asarray(extra, float)[np.isfinite(extra)]])
    lo, hi = vals.min(), vals.max()
    pad = 0.1 * (hi - lo if hi > lo else 1.0)
    ax.set_ylim(lo - pad, hi + pad)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("x")
    ax.grid(alpha=0.3)


def _cerrar(fig):
    plt.close(fig)
    return fig


def fig_area(g, a, b, exacto):
    """Volumen = área bajo la curva de la sección transversal A(x)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs, ys = _curva(g, a, b)
    ax.plot(xs, ys, color=VERDE, lw=2, label="A(x) = π([R−k]² − [r−k]²)")
    ax.fill_between(xs, ys, 0, color=VERDE, alpha=0.3, interpolate=True)
    ax.axvline(a, color="gray", ls="--", lw=1)
    ax.axvline(b, color="gray", ls="--", lw=1)
    _ejes(ax, ys)
    ax.set_ylabel("A(x)")
    ax.set_title(f"V = área bajo A(x) = {fmt(exacto)} u³")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _cerrar(fig)


def fig_metodo(m, g, a, b, exacto):
    """Aproximación del área bajo A(x): rectángulos (discos), trapecios o parábolas."""
    nombre, n, h, xs, ys = m["nombre"], m["n"], m["h"], m["xs"], m["ys"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cx, cy = _curva(g, a, b)
    ax.plot(cx, cy, color=VERDE, lw=2, label="A(x)")
    dibujado = n <= 100
    kw = dict(alpha=0.4, color=NARANJA, edgecolor=ROJO, lw=0.8)

    if dibujado:
        if nombre in METODOS_DISCO:
            for i in range(n):
                if np.isfinite(ys[i]):
                    ax.bar(a + i * h, ys[i], width=h, align="edge", **kw)
        elif nombre == "Trapecio":
            for i in range(n):
                if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]):
                    ax.fill([xs[i], xs[i], xs[i + 1], xs[i + 1]], [0, ys[i], ys[i + 1], 0], **kw)
        elif nombre == "Simpson 1/3":
            for k in range(0, n, 2):
                px, py = xs[k:k + 3], ys[k:k + 3]
                if np.all(np.isfinite(py)):
                    c = np.polyfit(px, py, 2)
                    t = np.linspace(px[0], px[2], 60)
                    ax.fill_between(t, np.polyval(c, t), 0, **kw)
        if n <= 50:
            ax.scatter(xs, ys, s=14, color=ROJO, zorder=3, label="puntos evaluados")
    err = abs(m["resultado"] - exacto)
    extra = "" if dibujado else "  (n > 100: no se dibujan las figuras)"
    ax.set_title(f"{nombre} · n = {n} · V ≈ {fmt(m['resultado'])} · error = {fmt(err)}{extra}", fontsize=10)
    _ejes(ax, cy, extra=ys if dibujado else ())
    ax.set_ylabel("A(x)")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return _cerrar(fig)


def fig_convergencia(g, a, b, exacto):
    """Error absoluto vs número de subdivisiones (escala log-log)."""
    ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre in METODOS:
        errs = np.array([abs(aproximar(g, a, b, n, nombre) - exacto) for n in ns], dtype=float)
        errs = np.where(errs < 1e-17, 1e-17, errs)
        ax.loglog(ns, errs, marker="o", ms=4, label=nombre)
    ax.set_xlabel("n (número de discos / subdivisiones)")
    ax.set_ylabel("error absoluto")
    ax.set_title("Convergencia: ¿qué método se acerca más rápido al volumen exacto?")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _cerrar(fig)