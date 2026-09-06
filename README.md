# Simulaciones interactivas de las Ecuaciones de Maxwell

Simulaciones numéricas e interactivas (Python + NumPy + Matplotlib) que
reconstruyen computacionalmente las cuatro ecuaciones de Maxwell.

> **Autores:** Juan Pablo Diaz Gamboa -- Justin Gabriel Lozano Núñez
> **Basado en:** *Computational electrodynamics training guide — Numerical
> analysis and simulation of Maxwell's equations* (Luz Esther González
> Reyes, Departamento de Física, Universidad del Tolima).

---

## 1. Contexto y objetivo

La guía de la actividad plantea que, para cada una de las cuatro ecuaciones
de Maxwell, se reconstruya computacionalmente **una (1)** de las
configuraciones físicas propuestas, siguiendo seis etapas: formulación
matemática → implementación numérica → representación gráfica → variación
de parámetros → interpretación física → relevancia tecnológica.

Este repositorio contiene los programas hechos para esa actividad, cada
uno enfocado en una de las ecuaciones. La idea no es solo mostrar una
fórmula, sino **generar** el fenómeno físico con código y poder
manipularlo en vivo (con sliders, botones y checkboxes) para observar cómo
cambia el campo cuando cambian sus fuentes.

## 2. Estado del repositorio

| # | Ecuación | Configuración elegida | Script | 
|---|----------|------------------------|--------|
| 1 | Ley de Gauss (electricidad), ∇·E = ρ/ε₀ | (b) Dipolo eléctrico (con modo carga puntual incluido) | [`dipolo.py`](./dipolo.py) |
| 2 | Ley de Gauss (magnetismo), ∇·B = 0 | (b) Espira circular | [`gauss_magnetico.py`](./gauss_magnetico.py) |
| 3 | Ley de Faraday, ∇×E = -∂B/∂t | (c) Transformador ideal (geometría de núcleo tipo ventana)| [`faraday.py`](./faraday.py) | 
| 4 | Ley de Ampère-Maxwell, ∇×B = μ₀J + μ₀ε₀∂E/∂t | (c) Pulso electromagnético propagándose (también sirve de ejemplo de la sección 7, "Full-system integration") | [`ampere_maxwell.py`](./ampere maxwell.py) | 

La sección 7 ("Full-system integration", derivar la ecuación de onda y verificar c = 1/√(μ₀ε₀)) queda cubierta por ampere_maxwell.py, que resuelve justamente esa ecuación de onda 1D.


## 3. Requisitos e instalación

```bash
pip install numpy matplotlib
```

Ambos scripts usan **widgets interactivos de Matplotlib** (sliders,
botones, checkboxes), por lo que necesitan un backend gráfico interactivo
(ventana propia), no el backend "inline" usado por defecto en algunos
notebooks de Jupyter.

- Si vas a correrlos desde una terminal normal, no necesitas hacer nada
  extra en la mayoría de los sistemas (usan TkAgg o QtAgg automáticamente).
- Si los corres desde Jupyter, ejecuta `%matplotlib widget` o
  `%matplotlib qt` en una celda **antes** de correr el script.
- Si aparece una ventana en blanco o un error de backend:
  ```bash
  pip install PyQt5
  # o, en Linux:
  sudo apt install python3-tk
  ```

## 4. Uso

```bash
python dipolo.py            # Ley de Gauss — electricidad
python gauss_magnetico.py   # Ley de Gauss — magnetismo
python faraday.py           # Ley de Faraday — transformador ideal
python ampere_maxwell.py    # Ley de Ampère-Maxwell — pulso EM 1D (FDTD)
```

### 4.1. `dipolo.py` — Ley de Gauss para la electricidad

Explorador 2D con cuatro paneles en tiempo real:

- **Campo eléctrico E(x,y)** (mapa vectorial),
- **Potencial V(x,y)** (mapa de color),
- **Curvas equipotenciales** V = cte.,
- **Divergencia numérica de E** (verifica que ∇·E ≈ 0 fuera de las cargas).

Controles:
- Slider **q (nC)** — magnitud de la carga.
- Slider **d (m)** — separación entre las cargas del dipolo.
- Botones de radio **Dipolo / Carga puntual** — cambian la configuración
  de fuentes sin reiniciar el programa.
- Partículas animadas que fluyen siguiendo la dirección local de E.
- Lectura interactiva de |E| y V al pasar el mouse sobre el panel de campo.

### 4.2. `gauss_magnetico.py` — Ley de Gauss para el magnetismo

Explorador 3D de una espira circular con corriente:

- Campo **B** calculado con la ley de **Biot-Savart** (discretización de
  la espira en 220 segmentos, evaluación vectorizada con NumPy).
- **Líneas de campo** integradas con **Runge-Kutta de 4º orden (RK4)**
  hecho a mano, aprovechando la simetría axial del problema.
- Verificación numérica de **∇·B ≈ 0** (máximo, media y RMS de la
  divergencia calculada por diferencias finitas en 3D), confirmando
  computacionalmente la ausencia de monopolos magnéticos.

Controles:
- Slider **Corriente I** y slider **Radio R**.
- Checkboxes para mostrar/ocultar: vectores B, líneas de campo, flechas de
  corriente, esfera gaussiana de referencia, malla de puntos de
  evaluación, rotación automática de la cámara.
- Botón **Restablecer cámara** y botón **Verificar ∇·B**.
- Arrastrar el mouse sobre la escena 3D rota la cámara libremente (cuando
  la rotación automática está desactivada).

### 4.3. `faraday.py` — Ley de Faraday (transformador ideal)

A diferencia de los tres scripts anteriores, aquí el campo NO se calcula en cada punto del espacio: se usa un modelo de circuito magnético concentrado (el modelo estándar de ingeniería para transformadores). Un núcleo tipo "ventana" (dos columnas + dos yugos) se reduce a una reluctancia Rm; las bobinas primaria y secundaria, a inductancias L1, L2 acopladas por una mutua M. Se resuelve un sistema de 2 ecuaciones diferenciales (integradas con Euler implícito) para las corrientes i1(t), i2(t), de donde se obtiene el flujo Φ(t) y su derivada dΦ/dt — la cantidad que la ley de Faraday relaciona directamente con la fem inducida: ε = -N·dΦ/dt.

Las líneas de campo B (dentro del núcleo) y de E (anillos alrededor de cada columna) son curvas geométricas fijas cuya intensidad y dirección se modulan en tiempo real con los resultados del modelo de circuito — el brillo de E responde a |dΦ/dt| (no a Φ), y su sentido se invierte según el signo de dΦ/dt, aplicando explícitamente la ley de Lenz.

Controles:

Sliders f, V₀ (fuente en el primario), N₁, N₂ (vueltas de cada devanado), R_Carga (resistencia del secundario).
Botones ON/OFF: Campo B, Campo E, Bobinas, Núcleo, Rotación 3D.
Panel de datos con V1, V2, flujo Φ, dΦ/dt y las fem inducidas ε1, ε2.

### 4.4. `ampere_maxwell.py` — Ley de Ampère-Maxwell (pulso EM 1D)

Simula la propagación de un pulso electromagnético en 1D resolviendo las ecuaciones de Maxwell dependientes del tiempo con el método FDTD (Finite-Difference Time-Domain, algoritmo de Yee): se integran las dos leyes rotacionales acopladas, ∂Hy/∂t = -(1/μ0)·∂Ez/∂x (Faraday) y ∂Ez/∂t = (1/ε)·∂Hy/∂x (Ampère-Maxwell), sobre una malla intercalada ("staggered") de puntos para E y para H. Usa diferencias finitas de 4º orden en el interior del dominio (menor dispersión numérica que el esquema clásico de 2º orden), una fuente "suave" con envolvente tipo tanh, y una condición de frontera absorbente de Mur para que la onda salga del dominio sin reflejarse artificialmente.

Incluye una interfaz entre dos medios (vacío y un medio con permitividad relativa εr ajustable) para observar transmisión y reflexión parcial —el resultado de este script es, en esencia, la solución numérica de la ecuación de onda que se deriva de combinar Faraday y Ampère-Maxwell (sección 7 "Full-system integration" de la guía), y su velocidad de propagación en el vacío debe coincidir con c = 1/√(μ₀ε₀).

Controles:

Sliders f, E₀ (frecuencia y amplitud de la fuente) y εr (medio) (permitividad relativa del segundo medio).
Checkboxes: Campo eléctrico E, Campo magnético H, Vector de Poynting (E×H), Interfaz de medio, Rotación automática.
Botones Pausa y Reiniciar.
Panel de datos con λ, εr, velocidad e índice de refracción del medio, y los valores instantáneos de E y H.
  

## 5. Fundamento físico resumido

| Ecuación | Forma diferencial | Qué modela cada script |
|---|---|---|
| Gauss (electricidad) | ∇·E = ρ/ε₀ | El campo E "nace" en las cargas positivas y "muere" en las negativas; fuera de las cargas, ∇·E = 0. |
| Gauss (magnetismo) | ∇·B = 0 | No existen monopolos magnéticos: toda línea de campo B se cierra sobre sí misma. |
| Faraday | ∇×E = -∂B/∂t | Un flujo magnético variable induce un campo eléctrico rotacional (base de generadores e inducción). |
| Ampère-Maxwell | ∇×B = μ₀J + μ₀ε₀∂E/∂t | La corriente de conducción **y** la corriente de desplazamiento generan campo magnético; este término faltante fue la clave de Maxwell para predecir las ondas electromagnéticas. |

Ambos scripts implementados calculan el campo con la **expresión analítica**
de superposición de fuentes puntuales/discretizadas (más eficiente y exacta
que resolver la ecuación de Poisson por diferencias finitas para estas
geometrías simples), y luego calculan la **divergencia de forma puramente
numérica** (con `np.gradient` o diferencias centradas manuales) para
verificar explícitamente, con datos, que la ley de Gauss correspondiente se
cumple — que es el ejercicio central que pide la guía de la actividad.

## 6. Por qué se hizo así (decisiones de diseño)

- **NumPy vectorizado en vez de bucles `for` por punto:** permite recalcular
  campos sobre mallas de miles de puntos en tiempo real cada vez que se
  mueve un slider, algo imposible con bucles puros de Python a 25 fps.
- **"Ablandamiento" (softening) de 1/r² y 1/r³:** evita división por cero
  o valores extremos justo sobre una carga/el alambre, a costa de un
  pequeño error solo en un radio muy chico alrededor de la fuente.
- **Normalización por percentil (no por el máximo absoluto)** al escalar
  colores y flechas: evita que un solo pico de campo cerca de una fuente
  puntual sature toda la escala visual y oculte la estructura del resto
  del campo.
- **Artistas persistentes de Matplotlib** (`set_UVC`, `set_data`,
  `set_segments`, etc.) en vez de limpiar y redibujar los ejes en cada
  actualización: más eficiente y sin parpadeo.
- **RK4 en vez de Euler** para las líneas de campo magnético: necesario
  porque una línea de B debe cerrarse sobre sí misma con buena precisión;
  un método de menor orden acumula error visible en una curva cerrada.
 - **Euler implícito (backward Euler)** para el circuito del transformador (`faraday.py`): a diferencia de Euler explícito, es incondicionalmente estable para el sistema lineal de inductancias acopladas, lo cual importa porque el paso de tiempo del circuito es fijo y no se reajusta según la frecuencia elegida con el slider.
 - **Diferencias finitas de 4º orden + condición de frontera de Mur** (`ampere_maxwell.py`): el esquema de 4º orden reduce la dispersión numérica (que distintas frecuencias del pulso viajarían a distinta velocidad solo por el error de discretización) frente al esquema clásico de 2º orden de Yee; la condición de Mur evita que la onda se refleje artificialmente al llegar al borde del dominio simulado, que de otro modo se comportaría como una pared en vez de un espacio abierto.
 - **Modelo de circuito concentrado en vez de campos distribuidos** (`faraday.py`): para un núcleo de alta permeabilidad, el modelo estándar de ingeniería (reluctancia, inductancia mutua) predice correctamente tensiones, corrientes y flujo sin necesidad de resolver el campo B punto a punto dentro del hierro, y es el enfoque que se usa en la práctica para diseñar transformadores reales.

## 7. Limitaciones conocidas

- Los campos se modelan con fuentes idealizadas (carga puntual, alambre
  filamentar): no incluyen efectos de tamaño finito, materiales, ni
  retardo temporal (no son simulaciones dependientes del tiempo, salvo la
  animación puramente visual de partículas/cámara).
- La divergencia numérica nunca es exactamente cero por el error de
  truncamiento propio de las diferencias finitas centradas (orden *h²*);
  se reporta como "aproximadamente cero" comparado con la escala típica
  del campo, no como cero exacto.
- `faraday.py` usa un modelo de CIRCUITO CONCENTRADO (reluctancia, inductancias), válido para núcleos de alta permeabilidad donde el flujo queda esencialmente confinado; no calcula el campo B o E punto a punto dentro o fuera del núcleo, así que las líneas de campo que se ven son una representación idealizada (geometría fija, solo modulada en intensidad/dirección), no el resultado de resolver una ecuación de campo.
- `ampere_maxwell.py` simula solo una dimensión espacial (propagación a lo largo de un eje, con E y H en las otras dos direcciones); no captura efectos 2D/3D como difracción, ni incidencia oblicua en la interfaz entre medios (solo incidencia normal).   
- Los scripts requieren un backend gráfico interactivo local; no están
  pensados para ejecutarse "as-is" en un entorno sin interfaz gráfica
  (por ejemplo, un notebook en la nube sin `%matplotlib widget`).

## 8. Licencia

"Uso académico —
Universidad del Tolima"
