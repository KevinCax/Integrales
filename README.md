# Volumen de sólidos de revolución (método de discos y arandelas)

Proyecto de **Cálculo II**: programa que calcula de forma **numérica** el volumen de un sólido de revolución por el
método de discos, y explica el procedimiento paso a paso para quien no sabe nada de integrales.

**Software:** Python 3 + Streamlit (interfaz web), SymPy, NumPy, SciPy, Matplotlib, Plotly, Pandas.

## Cómo ejecutarlo

```
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Cómo usarlo

1. Elige un **ejemplo listo** o escribe tu propia función `y = f(x)`.
2. Indica el **eje de giro**:
   - *Horizontal*: el eje x (`y = 0`) u otra recta `y = k`.
   - *Vertical*: el eje y (`x = 0`) u otra recta `x = k`. El programa **despeja x en función de y** por ti.
3. Escribe los límites (`a` y `b`; en giro vertical son límites de **y**) y la cantidad de discos `n`.
4. (Opcional) Escribe una segunda función para que quede un hueco: método de **arandelas**.

| Escribes | Significa |
|---|---|
| `x^2` o `x**2` | x al cuadrado |
| `2x` o `2*x` | 2 por x |
| `sin(x)`, `cos(x)`, `e^x`, `ln(x)`, `sqrt(x)`, `abs(x)` | funciones comunes (con paréntesis) |
| `pi`, `e` | constantes (también en los límites: `pi/2`) |
| `1,5` o `1.5` | decimales |

## Qué muestra el programa

1. **La idea, sin complicaciones:** qué es un sólido de revolución y por qué se corta en discos (con el sólido en 3D interactivo).
2. **Plantear el problema:** radios R y r, despeje de x si el eje es vertical, región 2D.
3. **Solución numérica (la parte principal):** una pestaña por método (discos con radio a la izquierda, derecha y al centro, trapecio y Simpson). Cada una con Δ, fórmula, sustitución, tabla disco por disco, gráfica 3D de los discos y gráfica de la aproximación.
4. **Valor exacto con la integral:** antiderivada → evaluar en los límites → restar, como se haría a mano, para comprobar la aproximación.
5. **Comparación:** tabla de errores y gráfica de convergencia (error vs n).
6. **Glosario** para principiantes.

## Casos especiales que maneja

- Giro alrededor de rectas horizontales y verticales, con discos o arandelas (la curva exterior se detecta sola).
- Invierte `y = f(x)` automáticamente; avisa si la función no se puede invertir en el intervalo.
- Singularidades removibles como `sin(x)/x` en x = 0 (usa el límite).
- Integrales divergentes o con asíntotas en el intervalo (`1/x` en [-1, 1]): mensaje claro.
- Funciones sin antiderivada elemental (`x^x`, `exp(x^2)`...): usa `scipy.integrate.quad` como valor de referencia.
- Entradas mal escritas o peligrosas: se rechazan con un mensaje en español (solo se permiten funciones y constantes conocidas).
- Simpson necesita n par: si n es impar usa n + 1.

## Estructura

```
app.py            interfaz web (Streamlit) con la explicación paso a paso
calculo.py        lógica matemática: lectura segura de funciones, inversión, exacta, numérica y LaTeX
graficos.py       gráficas 2D/3D (Plotly) y de aproximación/convergencia (Matplotlib)
test_calculo.py   pruebas de la lógica (python test_calculo.py)
requirements.txt  librerías necesarias
```
