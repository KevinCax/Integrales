"""graficos.py - Gráficas del proyecto (Matplotlib)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from calculo import METODOS, aproximar, fmt

AZUL, NARANJA, ROJO, VERDE = "#1f77b4", "#ff9f43", "#d62728", "#2ca02c"


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
    ax.set_ylabel("f(x)")
    ax.grid(alpha=0.3)


def fig_area(f, a, b, exacto):
    """Región exacta: verde = área sobre el eje x, roja = área bajo el eje x."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xs, ys = _curva(f, a, b)
    ax.plot(xs, ys, color=AZUL, lw=2, label="f(x)")
    ax.fill_between(xs, ys, 0, where=ys >= 0, color=VERDE, alpha=0.3, interpolate=True, label="área sobre el eje x (+)")
    ax.fill_between(xs, ys, 0, where=ys < 0, color=ROJO, alpha=0.3, interpolate=True, label="área bajo el eje x (−)")
    ax.axvline(a, color="gray", ls="--", lw=1)
    ax.axvline(b, color="gray", ls="--", lw=1)
    _ejes(ax, ys)
    ax.set_title(f"Región de integración   ∫f(x)dx = {fmt(exacto)}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def fig_metodo(m, f, a, b, exacto):
    """Dibuja la aproximación del método: rectángulos, trapecios o parábolas."""
    nombre, n, h, xs, ys = m["nombre"], m["n"], m["h"], m["xs"], m["ys"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    cx, cy = _curva(f, a, b)
    ax.plot(cx, cy, color=AZUL, lw=2, label="f(x)")
    dibujado = n <= 100
    kw = dict(alpha=0.4, color=NARANJA, edgecolor=ROJO, lw=0.8)

    if dibujado:
        if nombre in ("Riemann izquierda", "Riemann derecha", "Punto medio"):
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
    ax.set_title(f"{nombre} · n = {n} · S = {fmt(m['resultado'])} · error = {fmt(err)}{extra}", fontsize=10)
    _ejes(ax, cy, extra=ys if dibujado else ())
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def fig_convergencia(f, a, b, exacto):
    """Error absoluto vs número de subdivisiones (escala log-log)."""
    ns = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for nombre in METODOS:
        errs = np.array([abs(aproximar(f, a, b, n, nombre) - exacto) for n in ns], dtype=float)
        errs = np.where(errs < 1e-17, 1e-17, errs)
        ax.loglog(ns, errs, marker="o", ms=4, label=nombre)
    ax.set_xlabel("n (subdivisiones)")
    ax.set_ylabel("error absoluto")
    ax.set_title("Convergencia: ¿qué método se acerca más rápido al valor exacto?")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
