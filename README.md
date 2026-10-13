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

Escribe el enunciado tal como viene en el ejercicio, en el cuadro de la barra lateral, y elige la cantidad de discos `n`.
Las partes se separan con comas, punto y coma o saltos de línea. Ejemplos:

```
y=(x-2)^2, x=0, y=0, eje x
y=x^3-x, y=0, eje x
y=e^(-x), x=1, y=1, eje y=2
y=|cos(x)|, y=0, 0<=x<=2pi, eje x
y=tan(x), y=0, x=pi/4, eje x
y=x^2, x=0, y=4, eje y
```

El programa halla solo los puntos de corte y los límites de integración. Si el eje es vertical, despeja x en función de y.
Si el enunciado no dice el eje, usa el eje x y lo avisa.

| Escribes | Significa |
|---|---|
| `y=x^2`, `y=0` | curva o recta horizontal |
| `x=0`, `x=3`, `0<=x<=2pi` | rectas verticales (límites) |
| `eje x`, `eje y`, `eje y=2`, `eje x=1` | eje de giro |
| `x^2` o `x**2` | x al cuadrado |
| `2x` o `2*x` | 2 por x |
| `sin(x)`, `cos x`, `e^x`, `ln(x)`, `sqrt(x)`, `abs(x)`, `|x|` | funciones comunes |
| `pi`, `e` | constantes |
| `1,5` o `1.5` | decimales |

## Qué muestra el programa

1. **Planteamiento:** puntos de corte y límites, despeje de x si el eje es vertical, radios R y r, región 2D y sólido 3D.
2. **Solución numérica:** una pestaña por método (discos izquierda, derecha y punto medio, trapecio y Simpson), con Δ, fórmula, sustitución, cálculo, tabla disco por disco y gráficas.
3. **Valor exacto con la integral:** antiderivada, evaluación en los límites y resta.
4. **Comparación:** tabla de errores y gráfica de convergencia (error vs n).

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
