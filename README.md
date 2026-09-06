# Simulaciones interactivas de las Ecuaciones de Maxwell

Simulaciones numéricas e interactivas (Python + NumPy + Matplotlib) que
reconstruyen computacionalmente las cuatro ecuaciones de Maxwell, como
actividad de la asignatura **[Nombre del curso]** — [Nombre del programa],
Universidad del Tolima.

> **Autor(es):** [Tu nombre]
> **Docente:** [Nombre del docente]
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

| # | Ecuación | Configuración elegida | Script | Estado |
|---|----------|------------------------|--------|--------|
| 1 | Ley de Gauss (electricidad), ∇·E = ρ/ε₀ | (b) Dipolo eléctrico (con modo carga puntual incluido) | [`dipolo.py`](./dipolo.py) | ✅ Listo |
| 2 | Ley de Gauss (magnetismo), ∇·B = 0 | (b) Espira circular | [`gauss_magnetico.py`](./gauss_magnetico.py) | ✅ Listo |
| 3 | Ley de Faraday, ∇×E = -∂B/∂t | *(a) flujo magnético senoidal / (b) imán en movimiento / (c) transformador ideal* | `faraday.py` | 🔜 Próximamente |
| 4 | Ley de Ampère-Maxwell, ∇×B = μ₀J + μ₀ε₀∂E/∂t | *(a) corriente DC / (b) capacitor cargándose / (c) pulso EM* | `ampere_maxwell.py` | 🔜 Próximamente |

Cuando se agreguen los scripts 3 y 4, esta tabla y las secciones 4 y 5 de
este README se deben actualizar con su descripción, controles y capturas,
siguiendo el mismo formato usado para los dos primeros.

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

## 7. Entregables de la actividad (según la guía)

Según la sección "Expected deliverables" de la guía, además del código se
debe entregar un informe/póster que incluya: (1) fundamento teórico,
(2) metodología numérica, (3) resultados gráficos, (4) interpretación,
(5) limitaciones del modelo y (6) relación con sistemas fotónicos. Este
repositorio cubre los puntos (2) y (3) (metodología numérica y resultados
gráficos/interactivos); los puntos (1), (4), (5) y (6) se desarrollan en el
informe/póster que acompaña esta entrega.

## 8. Limitaciones conocidas

- Los campos se modelan con fuentes idealizadas (carga puntual, alambre
  filamentar): no incluyen efectos de tamaño finito, materiales, ni
  retardo temporal (no son simulaciones dependientes del tiempo, salvo la
  animación puramente visual de partículas/cámara).
- La divergencia numérica nunca es exactamente cero por el error de
  truncamiento propio de las diferencias finitas centradas (orden *h²*);
  se reporta como "aproximadamente cero" comparado con la escala típica
  del campo, no como cero exacto.
- Los scripts requieren un backend gráfico interactivo local; no están
  pensados para ejecutarse "as-is" en un entorno sin interfaz gráfica
  (por ejemplo, un notebook en la nube sin `%matplotlib widget`).

## 9. Licencia

[Especificar aquí, por ejemplo MIT License, o "Uso académico —
Universidad del Tolima"]
