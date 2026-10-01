# Integral definida: área bajo la curva

Proyecto de **Cálculo II** — Tema 1: programa que calcula de forma numérica el área bajo la curva de una función.

**Software:** Python 3 + Streamlit (interfaz web), SymPy, NumPy, SciPy, Matplotlib, Pandas.

## Cómo ejecutarlo

   ```
   python -m pip install -r requirements.txt
   python -m streamlit run app.py
   ```

## Cómo usarlo

| Escribes | Significa |
|---|---|
| `x^2` o `x**2` | x al cuadrado |
| `2x` o `2*x` | 2 por x |
| `sin(x)`, `cos(x)`, `exp(x)`, `ln(x)`, `sqrt(x)`, `abs(x)` | funciones comunes |
| `pi`, `e` | constantes (también en los límites: `pi/2`) |

## Qué muestra el programa

1. **Planteamiento:** la integral escrita en notación matemática.
2. **Solución exacta:** antiderivada → Teorema Fundamental del Cálculo → evaluar F(b) y F(a) → restar. Con la gráfica de la región (verde: sobre el eje x, roja: bajo el eje x).
3. **Solución numérica (una pestaña por método):** Riemann izquierda, Riemann derecha, punto medio, trapecio y Simpson 1/3. Cada una con Δx, fórmula, sustitución, cálculo, tabla de valores y gráfica de la aproximación.
4. **Comparación:** tabla de errores contra el valor exacto y gráfica de convergencia (error vs n).

## Casos especiales que maneja

- Funciones que cambian de signo: muestra la integral definida y el área geométrica total (∫|f|).
- Singularidades removibles como `sin(x)/x` en x = 0 (usa el límite).
- Integrales divergentes o con asíntotas en el intervalo (`1/x` en [-1, 1], `tan(x)` en [0, 2]): avisa con un mensaje claro.
- Funciones sin antiderivada elemental (`x^x`, `exp(x^2)`...): usa `scipy.integrate.quad` como valor de referencia.
- Simpson necesita n par: si n es impar usa n + 1.

## Estructura

```
app.py            interfaz web (Streamlit)
calculo.py        lógica matemática: exacta, numérica y procedimiento en LaTeX
graficos.py       gráficas con Matplotlib
test_calculo.py   pruebas de la lógica (python test_calculo.py)
requirements.txt  librerías necesarias
```
