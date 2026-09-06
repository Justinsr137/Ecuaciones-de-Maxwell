# -*- coding: utf-8 -*-
"""
Ley de Gauss para el magnetismo -- Espira circular 3D (VERSIÓN INTERACTIVA)
=============================================================================
Este script corresponde a la sección 4 ("Gauss law for magnetism", ∇·B = 0)
de la actividad "Computational electrodynamics training guide". De las tres
configuraciones que propone la guía (a) alambre recto infinito, (b) espira
circular, (c) dipolo magnético, aquí se implementó la opción (b) ESPIRA
CIRCULAR, simulada en 3D (no en un corte 2D) porque el campo de una espira
solo tiene simetría axial, no simetría de traslación como el alambre recto,
así que una vista 3D es la que mejor comunica su topología real.

Objetivo de la guía (4.1): verificar computacionalmente la ausencia de
monopolos magnéticos, es decir, que las líneas de B siempre se cierran sobre
sí mismas y que ∇·B = 0 en cualquier punto del espacio (a diferencia de ∇·E,
que sí puede ser distinto de cero donde hay carga eléctrica).

Cómo se calcula el campo:
- El campo B se obtiene con la ley de Biot-Savart, discretizando la espira
  en 220 segmentos de corriente y sumando vectorialmente la contribución de
  cada uno (NumPy vectorizado sobre todos los segmentos a la vez).
- Las líneas de campo NO se obtienen de una fórmula cerrada (a diferencia
  del dipolo eléctrico del otro script): se integran numéricamente con el
  método de RUNGE-KUTTA DE 4º ORDEN (RK4), implementado a mano, resolviendo
  la ecuación diferencial dr/ds = B(r)/|B(r)| (un vector unitario tangente
  al campo en cada punto). Por simetría axial, basta con integrar una línea
  en el plano meridional (plano que contiene el eje de la espira) y luego
  rotarla alrededor de ese eje para obtener las demás.
- Al final (4.3) se calcula la divergencia de B con diferencias finitas
  centradas en 3D, exactamente como en el otro script pero en las tres
  direcciones (x, y, z), para verificar numéricamente que ∇·B ≈ 0.

Interactividad (equivalente a un panel de control HTML, aquí hecho con
matplotlib.widgets):
    * Slider     -> Corriente I (reescala la magnitud de B; no cambia su
                    dirección ni su topología)
    * Slider     -> Radio R de la espira (cambia la geometría de la fuente
                    y, por lo tanto, la forma de las líneas de campo)
    * Checkboxes -> Vectores B / Líneas de campo / Flechas de corriente /
                     Esfera gaussiana / Malla de evaluación / Rotación
                     automática (mostrar u ocultar cada capa visual)
    * Botón      -> Restablecer cámara (vuelve a la vista 3D inicial)
    * Botón      -> Verificar ∇·B (recalcula y muestra max/mean/rms de la
                    divergencia numérica sobre una malla de puntos)
    * Arrastrar con el mouse sobre el gráfico 3D -> rotar la cámara
      libremente (funciona siempre que "Rotación automática" esté
      desactivada, ya que esta última controla la cámara en cada frame).

Paleta: negro y rojo dominantes. Cada elemento del campo tiene un color
propio dentro de esa familia cálida para que se distingan a simple vista
(elección estética, sin significado físico):
  - Espira (fuente de corriente)........... granate oscuro
  - Flechas de corriente.................... rojo brillante
  - Líneas de campo B (RK4)................. crema/dorado claro
  - Vectores B en la malla dispersa......... naranja/dorado
  - Esfera gaussiana........................ granate translúcido

Requisitos:  pip install numpy matplotlib
Ejecutar:    python gauss_magnetico.py
Requiere un backend gráfico interactivo (por defecto en la mayoría de
instalaciones de escritorio: TkAgg, QtAgg, etc.). Si ves una ventana en
blanco o un error de backend, prueba:
    pip install PyQt5   # o
    sudo apt install python3-tk
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.animation as animation
import itertools

# ============================================================
# 0) PALETA -- negro y rojo dominantes, con acentos diferenciados
# ============================================================

BG = "#050303"           # fondo de la figura y de los ejes 3D
PANEL_BG = "#0A0404"      # fondo de paneles de widgets
LINE = "#5C1414"          # bordes / líneas divisorias
INK = "#F2E9E7"           # texto principal (crema casi blanco)
INK_DIM = "#B08383"       # texto secundario

LOOP_COLOR = "#8B0000"        # espira (estructura fija) -> granate oscuro
CURRENT_COLOR = "#FF3B30"     # flechas de corriente -> rojo brillante
FIELD_LINE_COLOR = "#F2C9A0"  # líneas de campo B (RK4) -> crema/dorado claro
VECTOR_COLOR = "#FF9416"      # vectores B en malla dispersa -> naranja/dorado
SPHERE_COLOR = "#8B0000"      # esfera gaussiana -> granate translúcido
EVAL_COLOR = "#F2E9E7"        # puntos de malla de evaluación -> crema
TITLE_COLOR = "#FF453A"       # título / acento principal -> rojo vivo

# ============================================================
# 1) ESTADO FÍSICO (equivalente a las variables I, R del HTML)
# ============================================================

state = {"I": 10.0, "R": 1.5}   # corriente [A] y radio de la espira [m]
MU0 = 4 * np.pi * 1e-7          # permeabilidad magnética del vacío [T·m/A]
SEGMENTS = 220                  # nº de segmentos de corriente en que se
                                 # discretiza la espira para Biot-Savart
                                 # (más segmentos = más precisión, más costo)
SOFTENING = 0.055                # ablandamiento (igual idea que en el script
                                 # eléctrico): evita que 1/dist³ diverja para
                                 # puntos de evaluación muy cercanos al hilo

# ============================================================
# 2) BIOT-SAVART
# ============================================================

def build_source(R):
    """
    Discretiza la espira circular de radio R (centrada en el origen, en el
    plano xy) en SEGMENTS tramos rectos pequeños, la representación
    numérica estándar de un conductor curvo para aplicar Biot-Savart.

    Devuelve dos arreglos (SEGMENTS, 3):
    - pts: la posición 3D del centro de cada segmento de corriente.
    - dl:  el vector elemento de corriente d~l de cada segmento (tangente
      a la espira en ese punto, con magnitud = longitud de arco del
      segmento = R·dphi). d~l apunta en la dirección de circulación de la
      corriente I, que es lo que define el sentido del campo B generado
      (regla de la mano derecha).
    """
    phi = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    dphi = 2 * np.pi / SEGMENTS
    pts = np.stack([R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], axis=1)
    tangent = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    dl = tangent * (R * dphi)
    return pts, dl

# Fuente inicial (se reconstruye cada vez que cambia R, ver `rebuild`)
SRC_PTS, SRC_DL = build_source(state["R"])

def B_field(r):
    """
    Campo magnético B(r) generado por la espira, evaluado con la ley de
    Biot-Savart discretizada:

        B(r) = (μ0·I / 4π) · Σ_i  [ dl_i × (r - r_i) ] / |r - r_i|³

    donde la suma recorre los SEGMENTS segmentos de corriente construidos
    por `build_source` (r_i = SRC_PTS[i], dl_i = SRC_DL[i]).

    `r` puede ser un solo punto (vector de 3 componentes) o un arreglo de
    N puntos de forma (N, 3): `np.atleast_2d` normaliza el primer caso al
    segundo. El broadcasting de NumPy (`r[:, None, :] - SRC_PTS[None, :, :]`)
    calcula de una sola vez, sin bucles de Python, la diferencia entre
    CADA punto de evaluación y CADA segmento de la espira, resultando en
    un arreglo (N, SEGMENTS, 3); lo mismo con el producto cruz `np.cross`
    y la suma final sobre el eje de los segmentos (axis=1). Esta
    vectorización es la que permite evaluar B en toda una malla 3D o en
    todos los pasos de una integración RK4 con buen desempeño.

    `dist2` se ablanda sumando SOFTENING² (misma idea que SOFT en el
    script eléctrico) para que un punto de evaluación muy cercano al
    alambre no produzca una división por un número casi cero.
    """
    r = np.atleast_2d(r)
    diff = r[:, None, :] - SRC_PTS[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1) + SOFTENING**2
    dist3 = dist2 ** 1.5
    cross = np.cross(SRC_DL[None, :, :], diff)
    contrib = cross / dist3[..., None]
    B = (MU0 * state["I"] / (4 * np.pi)) * np.sum(contrib, axis=1)
    return B if B.shape[0] > 1 else B[0]

# ============================================================
# 3) RK4 PARA LÍNEAS DE CAMPO (dr/ds = B(r)/|B(r)|)
# ============================================================
# Una "línea de campo" es, por definición, una curva r(s) tal que en cada
# punto es tangente a B, es decir, satisface la ecuación diferencial
# ordinaria dr/ds = B(r)/|B(r)| (se usa el vector UNITARIO de B porque
# solo interesa la dirección de la curva, no qué tan rápido se recorre).
# A diferencia del dipolo eléctrico (donde la trayectoria de las
# partículas se aproxima con un simple paso de Euler), aquí se integra
# esta EDO con Runge-Kutta de 4º orden (RK4), un método clásico bastante
# más preciso que Euler para el mismo paso h, lo cual importa porque una
# línea de campo de B debe cerrarse sobre sí misma (no hay monopolos): un
# método menos preciso acumularía error y la curva "se abriría" en vez de
# formar un lazo cerrado alrededor de la espira.

def unit_tangent(r):
    """Vector tangente unitario a la línea de campo en el punto r:
    simplemente B(r) normalizado. Si B es (numéricamente) cero, se
    devuelve el vector nulo para no dividir por cero."""
    b = B_field(r)
    n = np.linalg.norm(b)
    return b / n if n > 1e-14 else np.zeros(3)

def rk4_step(r, h):
    """
    Un paso de Runge-Kutta de 4º orden para dr/ds = f(r), con f = unit_tangent.
    RK4 evalúa la función en 4 puntos distintos dentro del paso (k1 al
    principio, k2 y k3 en el punto medio con dos estimaciones distintas,
    k4 al final) y combina esas 4 pendientes con los pesos clásicos
    (1, 2, 2, 1)/6 para obtener una aproximación de orden 4 en h, mucho
    más precisa que un simple paso de Euler (orden 1) para el mismo costo
    de evaluar la función 4 veces en vez de 1.
    """
    k1 = unit_tangent(r)
    k2 = unit_tangent(r + 0.5 * h * k1)
    k3 = unit_tangent(r + 0.5 * h * k2)
    k4 = unit_tangent(r + h * k3)
    return r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def trace_meridional_line(x0, z0, h, max_steps, R):
    """
    Integra una línea de campo completa a partir de un punto semilla
    (x0, 0, z0) en el PLANO MERIDIONAL (el plano y=0 que contiene el eje
    de la espira). Por la simetría axial del problema, basta con trazar
    la línea en este único plano: las líneas en cualquier otro plano que
    contenga el eje son idénticas, solo rotadas (ver `rotate_z` y
    `compute_field_lines`), así que no hace falta repetir la integración
    para cada ángulo azimutal.

    La integración se detiene (criterio de parada) cuando la línea se
    aleja demasiado de la espira (> 3.2·R, ya escapó de la región de
    interés) o cuando se acerca demasiado al eje/centro (< 0.12·R, donde
    numéricamente se vuelve inestable), o tras `max_steps` pasos como
    límite de seguridad. Se fuerza `r[1] = 0.0` en cada paso para
    mantener la trayectoria exactamente en el plano y=0 (RK4 en 3D podría
    acumular un error numérico minúsculo fuera del plano).
    """
    r = np.array([x0, 0.0, z0])
    pts = [r.copy()]
    for _ in range(max_steps):
        if np.linalg.norm(r) > 3.2 * R or np.linalg.norm(r) < 0.12 * R:
            break
        r = rk4_step(r, h)
        r[1] = 0.0
        if not np.all(np.isfinite(r)):
            break
        pts.append(r.copy())
    return np.array(pts)

def rotate_z(points, angle):
    """Rotación rígida alrededor del eje z (el eje de simetría de la
    espira) por el ángulo dado, usando la matriz de rotación estándar.
    Se usa para generar, a partir de UNA línea meridional calculada con
    RK4, las copias rotadas que dan la apariencia 3D completa del haz de
    líneas de campo alrededor de la espira."""
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ Rz.T

# Puntos semilla (x, z) desde donde se lanza cada línea meridional,
# calibrados visualmente para un radio de referencia R=1.5 m (por eso se
# reescalan proporcionalmente a R en `compute_field_lines` mediante `scale`)
SEEDS_REF = [(1.0, 0.65), (1.25, 0.85), (1.55, 1.05),
             (1.85, 1.25), (2.2, 1.45), (2.65, 1.7)]      # calibrados para R=1.5
AZIMUTHS = np.linspace(0, 2 * np.pi, 8, endpoint=False)   # 8 copias rotadas
                                                            # equiespaciadas
                                                            # alrededor del eje

def compute_field_lines(R):
    """Genera el conjunto completo de líneas de campo en 3D: para cada
    semilla de referencia (reescalada al radio R actual) se integra una
    línea meridional con RK4 (`trace_meridional_line`) y luego se replica
    en las 8 direcciones azimutales con `rotate_z`, produciendo
    len(SEEDS_REF) * len(AZIMUTHS) líneas en total."""
    scale = R / 1.5
    h = 0.03 * R   # paso de integración proporcional a R (mismo nº de
                   # pasos relativos independientemente del tamaño de la espira)
    lines = []
    for x, z in SEEDS_REF:
        ml = trace_meridional_line(x * scale, z * scale, h, 260, R)
        for az in AZIMUTHS:
            lines.append(rotate_z(ml, az))
    return lines

# ============================================================
# 4) VERIFICACIÓN NUMÉRICA DE ∇·B  (sección 4.3 de la guía)
# ============================================================

def divergence_stats(R, n=9, lim=2.8, exclude=0.28):
    """
    Calcula estadísticas (máximo, media y RMS) de |∇·B| sobre una malla
    cúbica de n×n×n puntos en el cubo [-lim, lim]³, usando diferencias
    finitas centradas en las tres direcciones:

        ∇·B ≈ (Bx(x+h)-Bx(x-h))/(2h) + (By(y+h)-By(y-h))/(2h)
                                       + (Bz(z+h)-Bz(z-h))/(2h)

    Esto es el análogo en 3D del panel de divergencia del script
    eléctrico, pero en vez de un mapa de color 2D se resume en tres
    números, que son justo los que pide la guía ("compute the numerical
    divergence and discuss discretization errors").

    Los puntos demasiado cercanos al propio anillo de corriente
    (|dist_radial - R| < exclude, en el plano z=0) se excluyen del
    cálculo: igual que en el script eléctrico, la espira es una fuente
    de corriente idealizada (una curva 1D con densidad de corriente tipo
    delta), así que evaluar ahí encima produce valores enormes que no
    reflejan el comportamiento suave de ∇·B en el resto del espacio y
    dominarían las estadísticas de forma engañosa.

    El resultado esperado es que max/mean/rms sean pequeños en términos
    relativos (no exactamente cero, por el error de truncamiento O(h²)
    propio de las diferencias centradas), confirmando numéricamente que
    ∇·B = 0 como predice la ley de Gauss para el magnetismo.
    """
    h = 2 * lim / (n - 1)
    xs = np.linspace(-lim, lim, n)
    vals = []
    for x in xs:
        for y in xs:
            if abs(np.hypot(x, y) - R) < exclude:
                continue
            for z in xs:
                px, mx = B_field([x + h, y, z]), B_field([x - h, y, z])
                py, my = B_field([x, y + h, z]), B_field([x, y - h, z])
                pz, mz = B_field([x, y, z + h]), B_field([x, y, z - h])
                div = ((px[0]-mx[0]) + (py[1]-my[1]) + (pz[2]-mz[2])) / (2*h)
                if np.isfinite(div):
                    vals.append(abs(div))
    vals = np.array(vals)
    return vals.max(), vals.mean(), np.sqrt(np.mean(vals**2))

# ============================================================
# 5) FIGURA, EJES 3D Y WIDGETS
# ============================================================
# A diferencia del script eléctrico (2D, con 4 paneles), aquí se usa un
# solo eje 3D (`projection="3d"` de mplot3d) porque lo que se quiere
# comunicar es la TOPOLOGÍA cerrada de las líneas de B alrededor de la
# espira, algo que se aprecia mejor rotando la escena en el espacio que
# con cortes 2D separados.
fig = plt.figure(figsize=(18, 16), facecolor=BG)
ax = fig.add_axes([0.03, 0.32, 0.94, 0.56], projection="3d", facecolor=BG)
ax.set_axis_off()
LIM = 3.0
ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM); ax.set_zlim(-LIM, LIM)
ax.set_box_aspect([1, 1, 1])
ax.view_init(elev=22, azim=40)

title_text = ax.set_title("", color=TITLE_COLOR, fontsize=20, pad=22)
stats_line = ["max|∇·B| = —    mean|∇·B| = —    rms|∇·B| = —"]

def refresh_title():
    I, R = state["I"], state["R"]
    title_text.set_text(
        "Ley de Gauss para el magnetismo   ∇·B = 0\n"
        f"Espira circular (I = {I:.1f} A, R = {R:.2f} m) — líneas de campo con RK4\n"
        f"{stats_line[0]}"
    )

# --- Espira (se actualiza con set_data_3d, no se recrea) ---
# Igual que en el script eléctrico, se prefieren artistas persistentes que
# se actualizan in situ cuando es posible, en vez de recrearlos cada vez.
loop_line, = ax.plot([], [], [], color=LOOP_COLOR, lw=3, zorder=5)

# --- Colección única para TODAS las líneas de campo (rápida de actualizar) ---
# Line3DCollection permite guardar decenas de líneas 3D (una por cada
# combinación semilla × ángulo azimutal) como un solo objeto gráfico y
# reemplazarlas todas de una vez con `set_segments`, en vez de crear/borrar
# decenas de objetos `ax.plot` individuales en cada actualización.
field_line_collection = Line3DCollection([], colors=FIELD_LINE_COLOR, linewidths=1.1, alpha=0.78)
ax.add_collection3d(field_line_collection, autolim=False)

# --- Artistas que SÍ se recrean en cada rebuild (quiver 3D no es "editable") ---
# A diferencia de `loop_line` o `field_line_collection`, los objetos que
# devuelve `ax.quiver(...)` en 3D no tienen un método para cambiar sus
# datos una vez creados, así que hay que guardarlos en una lista mutable
# (para poder hacer `.remove()` del anterior) y volver a crearlos en cada
# llamada a `rebuild()`.
current_arrows_artist = [None]
vector_field_artist = [None]

# --- Elementos estáticos (no dependen de I ni R) ---
# Esfera gaussiana de referencia: una superficie cerrada imaginaria de
# radio Rs alrededor de la espira. Se muestra opcionalmente (checkbox)
# solo como ayuda visual/pedagógica para recordar el enunciado de la ley
# de Gauss en forma integral (∮ B·dA = 0 sobre cualquier superficie
# cerrada); el script NO integra B sobre esta esfera, solo la dibuja.
u, v = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 20))
Rs = 2.6
gaussian_surface = ax.plot_surface(
    Rs*np.cos(u)*np.sin(v), Rs*np.sin(u)*np.sin(v), Rs*np.cos(v),
    color=SPHERE_COLOR, alpha=0.09, linewidth=0
)
gaussian_surface.set_visible(False)

# Malla de puntos de evaluación en el plano z=0: se muestra opcionalmente
# para visualizar en qué puntos del espacio se calcula el campo B cuando
# se activa la capa "Vectores B" (ver `rebuild`), como referencia visual
# de la discretización usada.
grid_n, grid_lim = 10, 2.8
gx, gy = np.meshgrid(np.linspace(-grid_lim, grid_lim, grid_n),
                      np.linspace(-grid_lim, grid_lim, grid_n))
eval_points = ax.scatter(gx.ravel(), gy.ravel(), np.zeros(grid_n*grid_n),
                          s=6, color=EVAL_COLOR)
eval_points.set_visible(False)

# ============================================================
# 6) REBUILD (equivalente a rebuild() del HTML)
# ============================================================

def rebuild(_=None):
    """
    Función central que reconstruye TODA la escena 3D cada vez que cambia
    I, R, o se abre el programa por primera vez. Se llama desde los
    callbacks de los sliders (`on_I`, `on_R`) y al final del script.
    El argumento `_` se ignora: existe solo porque los widgets de
    matplotlib (Slider.on_changed) siempre pasan el nuevo valor como
    argumento al callback, aunque aquí no se necesite (se lee directo de
    `state`).
    """
    I, R = state["I"], state["R"]
    global SRC_PTS, SRC_DL
    # La corriente I NO afecta la geometría de la fuente, solo su
    # magnitud (ver B_field, que multiplica por state["I"]); por eso solo
    # es necesario reconstruir SRC_PTS/SRC_DL cuando cambia R.
    SRC_PTS, SRC_DL = build_source(R)

    # --- Espira: solo cambia su radio, así que se actualiza in situ ---
    phi = np.linspace(0, 2*np.pi, 240)
    loop_line.set_data(R*np.cos(phi), R*np.sin(phi))
    loop_line.set_3d_properties(np.zeros_like(phi))

    # --- Líneas de campo: se recalculan por completo con RK4 para el
    # nuevo radio R y se cargan de una vez en la colección persistente ---
    lines = compute_field_lines(R)
    field_line_collection.set_segments(lines)

    # --- Flechas de corriente (recreadas: los quiver 3D no tienen "set_data") ---
    # Se dibujan 8 flechas tangentes a la espira, espaciadas uniformemente
    # en ángulo, para indicar visualmente el sentido de circulación de I
    # (regla de la mano derecha: la dirección de estas flechas determina
    # hacia dónde apunta B dentro de la espira).
    if current_arrows_artist[0] is not None:
        current_arrows_artist[0].remove()
    arrow_phi = np.linspace(0, 2*np.pi, 8, endpoint=False)
    px = R*np.cos(arrow_phi); py = R*np.sin(arrow_phi); pz = np.zeros_like(arrow_phi)
    tx = -np.sin(arrow_phi)*0.35; ty = np.cos(arrow_phi)*0.35; tz = np.zeros_like(arrow_phi)
    current_arrows_artist[0] = ax.quiver(px, py, pz, tx, ty, tz,
                                          color=CURRENT_COLOR, linewidth=2, arrow_length_ratio=0.55)
    current_arrows_artist[0].set_visible(check.get_status()[2])

    # --- Vectores B en malla dispersa (recreados) ---
    # Se evalúa B(r) (con la I y R actuales) en un conjunto de puntos
    # distribuidos sobre dos ejes (x e y) y dos alturas (b = ±1), evitando
    # los puntos demasiado cercanos al radio de la espira (para no
    # dibujar un vector gigante justo sobre el alambre). La longitud de
    # cada flecha se comprime con una tangente hiperbólica (`np.tanh`)
    # respecto al vector de mayor magnitud, de forma que las flechas más
    # intensas no se salgan de una longitud razonable en pantalla, similar
    # en espíritu a la normalización por percentil del script eléctrico
    # pero adaptada a un conjunto de puntos mucho más pequeño.
    if vector_field_artist[0] is not None:
        vector_field_artist[0].remove()
    pts = []
    for a in np.linspace(-2.4, 2.4, 5):
        for b in [-1.0, 1.0]:
            for (px_, py_) in [(a, 0.0), (0.0, a)]:
                if np.hypot(px_, py_) - R != 0 and abs(np.hypot(px_, py_) - R) < 0.35:
                    continue
                pts.append([px_, py_, b])
    pts = np.array(pts)
    Bg = B_field(pts)
    Bn = np.linalg.norm(Bg, axis=1, keepdims=True)
    Bd = Bg / np.clip(Bn, 1e-15, None)
    scl = 0.5*np.tanh(Bn/np.max(Bn)*3)
    vector_field_artist[0] = ax.quiver(
        pts[:,0], pts[:,1], pts[:,2],
        Bd[:,0]*scl[:,0], Bd[:,1]*scl[:,0], Bd[:,2]*scl[:,0],
        color=VECTOR_COLOR, linewidth=1.0, alpha=0.8, arrow_length_ratio=0.35
    )
    vector_field_artist[0].set_visible(check.get_status()[0])

    field_line_collection.set_visible(check.get_status()[1])

    refresh_title()
    fig.canvas.draw_idle()

# ============================================================
# 7) WIDGETS: sliders, checkboxes, botones
# ============================================================

ax_I = fig.add_axes([0.19, 0.20, 0.38, 0.03])   # más a la derecha para que quepa la letra grande
ax_R = fig.add_axes([0.19, 0.15, 0.38, 0.03])
ax_I.set_facecolor(LINE)
ax_R.set_facecolor(LINE)
s_I = Slider(ax_I, "Corriente I [A]", 1.0, 20.0, valinit=state["I"], valstep=0.5,
             color=CURRENT_COLOR)
s_R = Slider(ax_R, "Radio R [m]", 0.8, 2.4, valinit=state["R"], valstep=0.05,
             color=LOOP_COLOR)

for s in (s_I, s_R):                     # tamaño de letra de los sliders (nombre y valor)
    s.label.set_fontsize(18)
    s.valtext.set_fontsize(18)
    s.label.set_color(INK)
    s.valtext.set_color(INK)

def on_I(val):
    """Callback del slider de corriente: cambiar I solo reescala la
    magnitud de B en todo el espacio (ver B_field), la geometría de la
    espira y de las líneas de campo no cambia, pero se llama a `rebuild()`
    igual porque las flechas de los vectores B en la malla dispersa sí
    necesitan recalcularse (su longitud depende de |B|)."""
    state["I"] = val
    rebuild()

def on_R(val):
    """Callback del slider de radio: cambiar R sí modifica la geometría de
    la fuente (build_source) y por lo tanto todo lo demás: la espira, las
    líneas de campo trazadas con RK4, las flechas de corriente y los
    vectores B."""
    state["R"] = val
    rebuild()

s_I.on_changed(on_I)
s_R.on_changed(on_R)

# --- Checkboxes -----------------------------------------------------------
# El tamaño del CUADRO clicable y de la MARCA se fija al crear el widget, con
# frame_props / check_props (parámetro "s" = área en puntos^2, como en
# plt.scatter). Súbelo si lo sigues viendo chico; bájalo si se ven muy grandes
# o se superponen entre filas.
ax_check = fig.add_axes([0.64, 0.02, 0.34, 0.30])   # más ancho y alto para dar espacio
ax_check.set_facecolor(PANEL_BG)
for spine in ax_check.spines.values():
    spine.set_color(LINE)
labels = ["Vectores B", "Líneas de campo", "Flechas de corriente",
          "Esfera gaussiana", "Malla de evaluación", "Rotación automática"]
actives = [True, True, True, False, False, False]
check = CheckButtons(
    ax_check, labels, actives,
    label_props={"fontsize": [20], "color": [INK]},
    frame_props={"s": [220], "linewidth": [1.4], "edgecolor": [INK_DIM]},  # TAMAÑO DEL CUADRO
    check_props={"s": [170], "facecolor": [CURRENT_COLOR]},                # TAMAÑO DE LA MARCA
)

def on_check(label):
    """Callback único para las 6 casillas de verificación: cada casilla
    simplemente muestra/oculta (`set_visible`) la capa gráfica
    correspondiente. La casilla "Rotación automática" es la excepción: no
    controla ningún artista de dibujo, sino que su estado se consulta
    directamente dentro de `update_frame` (la función de la animación de
    cámara) en cada frame, así que aquí no requiere ninguna acción."""
    st = check.get_status()
    idx = labels.index(label)
    is_on = st[idx]
    if label == "Vectores B":
        vector_field_artist[0].set_visible(is_on)
    elif label == "Líneas de campo":
        field_line_collection.set_visible(is_on)
    elif label == "Flechas de corriente":
        current_arrows_artist[0].set_visible(is_on)
    elif label == "Esfera gaussiana":
        gaussian_surface.set_visible(is_on)
    elif label == "Malla de evaluación":
        eval_points.set_visible(is_on)
    # "Rotación automática" se lee directamente en update_frame()
    fig.canvas.draw_idle()

check.on_clicked(on_check)

ax_reset = fig.add_axes([0.12, 0.025, 0.22, 0.06])   # caja más alta/ancha para que quepa la letra grande
btn_reset = Button(ax_reset, "Restablecer cámara", color=PANEL_BG, hovercolor="#241010")
btn_reset.label.set_color(INK)
btn_reset.label.set_fontsize(20)
for spine in ax_reset.spines.values():
    spine.set_color(LINE)

def on_reset(_):
    """Restaura la cámara del gráfico 3D a la orientación inicial
    (elevación 22°, azimut 40°), útil después de haber arrastrado el
    mouse para rotar la vista libremente."""
    ax.view_init(elev=22, azim=40)
    fig.canvas.draw_idle()

btn_reset.on_clicked(on_reset)

ax_stats = fig.add_axes([0.37, 0.025, 0.24, 0.06])   # caja más alta/ancha
btn_stats = Button(ax_stats, "Verificar ∇·B", color=PANEL_BG, hovercolor="#241010")
btn_stats.label.set_color(INK)
btn_stats.label.set_fontsize(20)
for spine in ax_stats.spines.values():
    spine.set_color(LINE)

def on_stats(_):
    """Callback del botón 'Verificar ∇·B': dispara el cálculo (algo más
    costoso, del orden de n³ evaluaciones de B_field) de `divergence_stats`
    para el radio actual, y muestra el resultado en el título de la
    figura. No se recalcula automáticamente en cada frame ni en cada
    cambio de slider porque sería demasiado lento para uso interactivo en
    tiempo real; se deja como una acción explícita del usuario."""
    dmax, dmean, drms = divergence_stats(state["R"])
    stats_line[0] = f"max|∇·B| = {dmax:.2e}    mean|∇·B| = {dmean:.2e}    rms|∇·B| = {drms:.2e}"
    refresh_title()
    fig.canvas.draw_idle()

btn_stats.on_clicked(on_stats)

# ============================================================
# 8) ROTACIÓN AUTOMÁTICA (opcional) VÍA FuncAnimation
#     — si está desactivada, el mouse controla la cámara libremente.
# ============================================================

def update_frame(_frame):
    """
    Callback de la animación de cámara, llamado ~25 veces por segundo
    (interval=40 ms). Solo hace algo si la casilla "Rotación automática"
    (índice 5 en `labels`/`check.get_status()`) está activa: en ese caso
    incrementa el ángulo azimutal de la vista 0.6° por frame (≈15°/s),
    dando una rotación continua alrededor del eje vertical. Si la casilla
    está desactivada, esta función no hace nada y el usuario controla la
    cámara libremente arrastrando con el mouse (comportamiento nativo de
    los ejes 3D de matplotlib).

    `frames=itertools.count()` genera una animación de duración indefinida
    (un contador infinito 0, 1, 2, ...) en vez de un número fijo de frames.
    """
    if check.get_status()[5]:                 # "Rotación automática"
        az = (ax.azim + 0.6) % 360
        ax.view_init(elev=ax.elev, azim=az)
    return []

# Igual que `anim` en el script eléctrico, `ani` debe conservarse en una
# variable de módulo para que no se destruya por garbage collection.
ani = animation.FuncAnimation(fig, update_frame, frames=itertools.count(),
                               interval=40, blit=False, cache_frame_data=False)

# ============================================================
# 9) INICIALIZACIÓN
# ============================================================

rebuild()        # construye la escena 3D con los valores iniciales de I y R
on_stats(None)   # primer cálculo de divergencia al abrir, para que el
                 # título ya muestre max/mean/rms de ∇·B desde el arranque

if __name__ == "__main__":
    plt.show()