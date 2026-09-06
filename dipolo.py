"""
Ley de Gauss para la electricidad — Explorador interactivo del campo eléctrico
================================================================================
Este script corresponde a la sección 3 ("Gauss law for electricity",
∇·E = ρ/ε0) de la actividad "Computational electrodynamics training guide".
De las cuatro configuraciones que propone la guía (carga puntual, dipolo,
nube de carga gaussiana, capacitor de placas paralelas) aquí se implementó
la opción (b) DIPOLO, con la opción (a) CARGA PUNTUAL disponible como modo
alternativo dentro del mismo programa (se elige con los botones de radio).

En vez de resolver la ecuación de Poisson ∇²V = -ρ/ε0 con un método
numérico de diferencias finitas (que sería necesario para geometrías de
carga arbitrarias, como la nube gaussiana o el capacitor), aquí se aprovecha
que para cargas puntuales/dipolo la solución analítica de V y E ya se conoce
(principio de superposición de cargas puntuales). Esto permite evaluar el
campo y el potencial en cualquier punto de forma directa y vectorizada con
NumPy, sin necesidad de resolver un sistema lineal. La divergencia, en
cambio, sí se calcula numéricamente (con diferencias finitas centradas vía
`np.gradient`) para verificar explícitamente que ∇·E ≈ 0 fuera de las
cargas, que es el análisis que pide la sección 3.3 de la guía.

Qué muestra el programa (los 4 paneles cubren los 4 puntos de la sección
3.3 "Required analysis" de la guía):
- Panel superior izquierdo:  mapa vectorial del campo eléctrico E(x,y).
- Panel superior derecho:    mapa de color del potencial escalar V(x,y).
- Panel inferior izquierdo:  curvas equipotenciales V(x,y) = cte.
- Panel inferior derecho:    divergencia numérica ∇·E (debe ser ≈ 0 en
                              todo punto que no sea una carga puntual).

Controles interactivos (para el punto 4 "Parameter variation" de la guía):
- Slider "q (nC)": magnitud de la carga (o de cada carga del dipolo).
- Slider "d (m)":  separación entre las dos cargas del dipolo (se oculta
  en modo "Carga puntual", donde no aplica).
- Botones de radio "Dipolo" / "Carga puntual": cambian la configuración
  de fuentes sin reiniciar el programa.
- Animación de partículas: partículas que se desplazan siguiendo la
  dirección local de E, dando una intuición visual de las líneas de campo
  (actualizada ~25 veces por segundo con matplotlib.animation.FuncAnimation).
- Lectura interactiva: al mover el mouse sobre el panel de campo se
  muestran en vivo los valores numéricos de |E| y V en ese punto.

Requisitos:  pip install numpy matplotlib
Ejecutar:    python dipolo.py
(usa un backend interactivo, p. ej. TkAgg / Qt5Agg; en Jupyter usar
 `%matplotlib widget` o `%matplotlib qt` antes de correr el script, porque
 los sliders y botones no funcionan con el backend "inline" por defecto)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.animation import FuncAnimation

# ----------------------------------------------------------------------
# Constantes físicas y dominio
# ----------------------------------------------------------------------
EPS0 = 8.8541878128e-12          # permitividad eléctrica del vacío [F/m]
K_E = 1.0 / (4.0 * np.pi * EPS0)  # constante de Coulomb k = 1/(4π ε0) [N·m²/C²]

# "Ablandamiento" (softening): en vez de evaluar 1/r² o 1/r exactamente en
# r=0 (donde E y V divergen a infinito sobre la propia carga puntual),
# se reemplaza r por sqrt(r² + SOFT²). Esto es una técnica estándar en
# simulación de N-cuerpos/electrostática numérica: evita overflow/NaN y
# evita que un solo píxel cercano a la carga sature toda la escala de
# color, a costa de "redondear" el campo en un radio ~SOFT alrededor de
# cada carga (fuera de ese radio el resultado es físicamente exacto).
SOFT = 0.05  # [m]

# Región del plano (x, y) que se dibuja, en metros. Se eligió más ancha
# en x que en y porque el dipolo se alinea sobre el eje x y así se ve
# completo su patrón de campo característico.
XMIN, XMAX, YMIN, YMAX = -4.5, 4.5, -1.8, 1.8

# Estado global mutable de la simulación: qué configuración de cargas se
# está mostrando y con qué parámetros. Los sliders y los botones de radio
# modifican este diccionario; todas las funciones de física y dibujo leen
# de aquí, así que actualizar `state` y volver a llamar a `update_static()`
# es suficiente para refrescar toda la figura.
state = {"mode": "dipole", "q_nC": 1.0, "d_m": 2.0}


# ----------------------------------------------------------------------
# Física: modelo de cargas puntuales y superposición
# ----------------------------------------------------------------------
def get_charges():
    """
    Devuelve la lista de fuentes de carga [(q, x, y, etiqueta), ...] que
    corresponde al modo actual guardado en `state`.

    - Modo "point": una sola carga q en el origen (configuración (a) de
      la guía, "punto de carga").
    - Modo "dipole": dos cargas de igual magnitud y signo opuesto (+q y
      -q), separadas una distancia d y centradas en el origen sobre el
      eje x (configuración (b) de la guía, "electric dipole").

    q_nC está en nanoculombios en la interfaz (más intuitivo para mover
    con un slider) y aquí se convierte a culombios (×1e-9) para que el
    resto de las fórmulas físicas trabajen en unidades del SI.
    """
    q = state["q_nC"] * 1e-9
    if state["mode"] == "point":
        return [(q, 0.0, 0.0, "q")]
    d = state["d_m"]
    return [(q, -d / 2, 0.0, "+q"), (-q, d / 2, 0.0, "-q")]


def e_field(X, Y):
    """
    Campo eléctrico E(x,y) evaluado con el principio de superposición:
    para cada carga puntual q ubicada en (cx, cy), su contribución es
        E = k·q·(r - r_q) / |r - r_q|³
    (la forma vectorial de la ley de Coulomb), y el campo total es la
    suma de las contribuciones de todas las cargas de `get_charges()`.

    X, Y pueden ser arreglos de NumPy de cualquier forma (por ejemplo,
    una malla 2D generada con np.meshgrid): toda la función está
    vectorizada, es decir, evalúa el campo en TODOS los puntos de la
    malla a la vez con operaciones de NumPy, en lugar de recorrer punto
    por punto con un bucle de Python (esto es lo que permite refrescar
    los cuatro paneles en tiempo real cuando se mueve un slider).

    r2 se acota por abajo con SOFT² (ver comentario junto a la constante
    SOFT) para que 1/r³ nunca se dispare a infinito exactamente sobre
    una carga.
    """
    Ex = np.zeros_like(X, dtype=float)
    Ey = np.zeros_like(Y, dtype=float)
    for q, cx, cy, _ in get_charges():
        dx, dy = X - cx, Y - cy
        r2 = np.maximum(dx * dx + dy * dy, SOFT * SOFT)
        r3 = r2 * np.sqrt(r2)
        c = K_E * q / r3
        Ex += c * dx
        Ey += c * dy
    return Ex, Ey


def potential(X, Y):
    """
    Potencial eléctrico escalar V(x,y) = Σ k·q / r, obtenido igualmente
    por superposición de las contribuciones de cada carga puntual.

    V es la cantidad "primitiva" a partir de la cual se puede recuperar
    el campo como E = -∇V; en este script no se deriva V numéricamente
    para obtener E (eso sería el enfoque de diferencias finitas que pide
    la ecuación de Poisson, ∇²V = -ρ/ε0, de la guía) porque para cargas
    puntuales ya se tiene la expresión analítica cerrada de E. Se calcula
    V de forma independiente porque el panel de potencial y el panel de
    curvas equipotenciales lo necesitan directamente.
    """
    V = np.zeros_like(X, dtype=float)
    for q, cx, cy, _ in get_charges():
        dx, dy = X - cx, Y - cy
        r = np.maximum(np.sqrt(dx * dx + dy * dy), SOFT)
        V += K_E * q / r
    return V


# ----------------------------------------------------------------------
# Mallas de evaluación
# ----------------------------------------------------------------------
# Se usan TRES mallas distintas, cada una con la resolución justa para su
# propósito, en vez de una sola malla muy fina para todo. Esto es una
# decisión de rendimiento: recalcular una malla de 220×90 puntos en cada
# frame de la animación (25 fps) sería demasiado lento, mientras que una
# malla de solo 24×11 puntos no se ve bien como mapa de color continuo.

NX_HI, NY_HI = 220, 90          # malla "fina": potencial y divergencia
                                 # (necesitan buena resolución para que el
                                 # mapa de color y el gradiente numérico
                                 # np.gradient se vean suaves y precisos)
xs_hi = np.linspace(XMIN, XMAX, NX_HI)
ys_hi = np.linspace(YMIN, YMAX, NY_HI)
X_hi, Y_hi = np.meshgrid(xs_hi, ys_hi)

NX_EQ, NY_EQ = 140, 60          # malla intermedia: curvas equipotenciales
                                 # (matplotlib.contour necesita suficientes
                                 # puntos para trazar líneas suaves, pero
                                 # no tantos como el mapa de color)
xs_eq = np.linspace(XMIN, XMAX, NX_EQ)
ys_eq = np.linspace(YMIN, YMAX, NY_EQ)
X_eq, Y_eq = np.meshgrid(xs_eq, ys_eq)

NX_Q, NY_Q = 24, 11             # malla dispersa: flechas de campo (quiver)
                                 # una flecha por cada punto de esta malla;
                                 # si fuera muy densa las flechas se
                                 # encimarían y no se distinguirían.
xs_q = np.linspace(XMIN, XMAX, NX_Q)
ys_q = np.linspace(YMIN, YMAX, NY_Q)
X_q, Y_q = np.meshgrid(xs_q, ys_q)

# --- Partículas de "flujo" (no son una malla fija, sino puntos móviles) ---
# Estas partículas no forman parte de ningún cálculo físico: son un
# recurso puramente visual para sugerir la dirección de las líneas de
# campo mediante su movimiento (ver función `animate` más abajo).
N_PARTICLES = 160
rng = np.random.default_rng(7)  # semilla fija -> posiciones reproducibles


def respawn(n):
    """Genera n posiciones aleatorias uniformes dentro del dominio visible.
    Se usa tanto para la inicialización de las partículas como para
    "reciclar" las que salen del dominio o caen sobre una carga."""
    pts = np.column_stack(
        [rng.uniform(XMIN, XMAX, n), rng.uniform(YMIN, YMAX, n)]
    )
    return pts


particles = respawn(N_PARTICLES)

# ----------------------------------------------------------------------
# Figura y ejes  —  paleta negro / rojo
# ----------------------------------------------------------------------
# Esta sección es puramente estética (colores, tipografía, disposición de
# los 4 paneles); no tiene contenido físico. Se documenta igual porque es
# la parte que más se suele querer personalizar al reutilizar el script
# (por ejemplo, para que combine con los colores del póster o la
# presentación donde se muestren estas figuras).
BG = "#050303"
PANEL_BG = "#0A0404"
LINE = "#5C1414"
INK = "#F2E9E7"
INK_DIM = "#B08383"
CORAL = "#FF3B30"   # rojo brillante -> carga positiva, acentos, slider q
TEAL = "#8B0000"    # rojo oscuro/granate -> carga negativa, slider d
AMBER = "#FF453A"   # rojo de acento -> título, partículas de flujo

from matplotlib.colors import LinearSegmentedColormap

# Colormap para las flechas de campo: negro -> rojo -> naranja/dorado
# (deliberadamente distinto del rojo puro usado en potencial/equipotenciales)
CMAP_FIELD = LinearSegmentedColormap.from_list(
    "black_orange_seq", ["#000000", "#5C0A0A", "#C81E00", "#FF7A18", "#FFC145"]
)
# Colormap divergente granate -> negro -> rojo (para potencial y divergencia)
CMAP_DIV = LinearSegmentedColormap.from_list(
    "black_red_div", ["#8B0000", "#3A0000", "#000000", "#3A0000", "#FF3B30"]
)
# Colores fijos para las curvas equipotenciales: rojo (V>0) vs. crema (V<0)
EQUIP_POS = "#FF3B30"
EQUIP_NEG = "#F2C9A0"

plt.rcParams["font.family"] = "sans-serif"
fig = plt.figure(figsize=(13, 9), facecolor=BG)
# Rejilla 2x2: cada celda es uno de los 4 análisis requeridos en la
# sección 3.3 de la guía (mapa de potencial, mapa vectorial de E, curvas
# equipotenciales y divergencia numérica de E).
gs = fig.add_gridspec(
    2, 2, left=0.06, right=0.95, top=0.85, bottom=0.30, wspace=0.25, hspace=0.35
)
ax_field = fig.add_subplot(gs[0, 0])   # campo E (vectores/quiver)
ax_pot = fig.add_subplot(gs[0, 1])     # potencial V (mapa de color)
ax_equip = fig.add_subplot(gs[1, 0])   # curvas equipotenciales V=cte.
ax_div = fig.add_subplot(gs[1, 1])     # divergencia numérica de E
axes_all = [ax_field, ax_pot, ax_equip, ax_div]
titles = [
    "Campo eléctrico   E(r) = k·q·(r−r_q)/|r−r_q|³",
    "Potencial eléctrico   V(r) = Σ k·q/r",
    "Curvas equipotenciales   V(x,y) = cte.",
    "Divergencia numérica   ∇·E ≈ ∂Eₓ/∂x + ∂E_y/∂y",
]

fig.suptitle(
    "Ley de Gauss — campo y potencial eléctrico     ∇·E = ρ/ε₀",
    color=AMBER, fontsize=15, x=0.06, ha="left", family="monospace",
)

for ax, title in zip(axes_all, titles):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color=INK_DIM, fontsize=9.5, loc="left", family="monospace")
    ax.set_xlim(XMIN, XMAX)
    ax.set_ylim(YMIN, YMAX)
    ax.set_aspect("equal")
    ax.tick_params(colors=INK_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(LINE)

readout_text = fig.text(
    0.06, 0.885, "Mueve el mouse sobre el panel de campo para leer E y V.",
    color=TEAL, fontsize=10, family="monospace",
)

from matplotlib.lines import Line2D
equip_legend = ax_equip.legend(
    handles=[
        Line2D([0], [0], color=EQUIP_POS, lw=1.4, label="V > 0"),
        Line2D([0], [0], color=EQUIP_NEG, lw=1.4, ls="dashed", label="V < 0"),
    ],
    loc="upper right", fontsize=7.5, facecolor=PANEL_BG, edgecolor=LINE,
    labelcolor=INK_DIM, framealpha=0.85,
)

# ----------------------------------------------------------------------
# Artistas persistentes
# ----------------------------------------------------------------------
# "Artista persistente" = un objeto gráfico de matplotlib que se crea UNA
# sola vez y luego se actualiza en el sitio (con métodos como set_UVC,
# set_data, set_offsets) cada vez que cambia un slider, en lugar de borrar
# y volver a dibujar todo el eje desde cero. Esto es mucho más eficiente y
# evita el parpadeo típico de recrear gráficos en cada actualización.
quiv = ax_field.quiver(
    X_q, Y_q, np.zeros_like(X_q), np.zeros_like(Y_q), np.zeros_like(X_q),
    cmap=CMAP_FIELD, scale=18, width=0.0045, pivot="tail",
)
flow_scatter = ax_field.scatter([], [], s=4, c=AMBER, alpha=0.85, zorder=5)

pot_im = ax_pot.imshow(
    np.zeros((NY_HI, NX_HI)), extent=[XMIN, XMAX, YMIN, YMAX], origin="lower",
    cmap=CMAP_DIV, aspect="auto",
)
pot_cbar = fig.colorbar(pot_im, ax=ax_pot, fraction=0.046, pad=0.03)
pot_cbar.ax.tick_params(colors=INK_DIM, labelsize=7)

div_im = ax_div.imshow(
    np.zeros((NY_HI, NX_HI)), extent=[XMIN, XMAX, YMIN, YMAX], origin="lower",
    cmap=CMAP_DIV, aspect="auto",
)
div_cbar = fig.colorbar(div_im, ax=ax_div, fraction=0.046, pad=0.03)
div_cbar.ax.tick_params(colors=INK_DIM, labelsize=7)

equip_artist = {"pos": None, "neg": None}
charge_scatters = {ax: ax.scatter([], [], s=60, edgecolors=PANEL_BG, linewidths=1.2, zorder=6)
                    for ax in axes_all}
charge_labels = {ax: [] for ax in axes_all}


def draw_charge_markers():
    for ax in axes_all:
        chs = get_charges()
        xs = [c[1] for c in chs]
        ys = [c[2] for c in chs]
        colors = [CORAL if c[0] > 0 else TEAL for c in chs]
        charge_scatters[ax].set_offsets(np.column_stack([xs, ys]) if xs else np.empty((0, 2)))
        charge_scatters[ax].set_color(colors)
        for t in charge_labels[ax]:
            t.remove()
        charge_labels[ax] = []
        for c in chs:
            t = ax.text(c[1], c[2] + 0.15, c[3], color=INK, fontsize=8,
                         ha="center", family="monospace", zorder=7)
            charge_labels[ax].append(t)


def percentile_abs(arr, p):
    """
    Devuelve el percentil p (0-100) de |arr|, ignorando valores no finitos
    (inf/NaN, que pueden aparecer cerca de una carga a pesar del
    ablandamiento). Se usa para fijar el rango de color (vmin=-val,
    vmax=+val) de los mapas de potencial y divergencia: igual que en
    `update_field_panel`, usar un percentil alto (en vez del máximo
    absoluto) evita que el pico de valor sobre una carga puntual sature
    toda la escala de color y deje el resto del mapa plano/invisible.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    val = np.percentile(np.abs(finite), p)
    return val if val > 0 else 1.0


# ----------------------------------------------------------------------
# Actualización de los paneles "estáticos" (se recalculan al mover un slider)
# ----------------------------------------------------------------------
def update_field_panel():
    """
    Actualiza las flechas del panel de campo eléctrico (panel superior
    izquierdo). El campo eléctrico de una carga puntual varía como 1/r²,
    así que sus flechas cerca de la carga serían muchísimo más largas que
    lejos de ella si se dibujaran a escala real: el panel sería ilegible
    (unas pocas flechas gigantes y el resto invisibles).

    Para resolverlo:
    1. Se enmascaran (mask) los puntos demasiado cercanos a una carga
       (r <= 0.42), donde el campo es enorme y no aporta información útil.
    2. Se normaliza la magnitud de cada flecha respecto al percentil 90
       de |E| en el resto de puntos (`maxmag`), en vez de respecto al
       máximo absoluto: así una sola flecha extrema no aplasta la escala
       de todas las demás.
    3. Cada flecha se dibuja con dirección igual a la del campo real
       (Ex/|E|, Ey/|E|) pero con longitud recortada a [0, 1] mediante
       `rel`, y el color (C) también codifica esa magnitud relativa.
    Esto es una elección de VISUALIZACIÓN, no cambia el campo físico
    calculado por `e_field`, solo cómo se representa su magnitud.
    """
    Ex, Ey = e_field(X_q, Y_q)
    mag = np.hypot(Ex, Ey)
    mask = mag > 1e-16
    for q, cx, cy, _ in get_charges():
        r = np.hypot(X_q - cx, Y_q - cy)
        mask &= r > 0.42
    mag_safe = np.where(mask, mag, np.nan)
    maxmag = np.nanpercentile(mag_safe, 90) if np.any(mask) else 1.0
    if not np.isfinite(maxmag) or maxmag <= 0:
        maxmag = 1.0
    rel = np.clip(mag / maxmag, 0, 1)
    U = np.where(mask, Ex / np.maximum(mag, 1e-30) * rel, 0.0)
    V = np.where(mask, Ey / np.maximum(mag, 1e-30) * rel, 0.0)
    C = np.where(mask, rel, np.nan)
    quiv.set_UVC(U, V, C)
    quiv.set_clim(0, 1)


def update_potential_panel():
    """Evalúa V(x,y) en la malla fina y actualiza el mapa de color del
    panel de potencial. El colormap CMAP_DIV es divergente (negativo ->
    cero -> positivo) para que sea visualmente claro dónde V cambia de
    signo, algo importante en el dipolo porque V es negativo cerca de la
    carga -q y positivo cerca de +q."""
    Vgrid = potential(X_hi, Y_hi)
    vmax = percentile_abs(Vgrid, 96)
    pot_im.set_data(Vgrid)
    pot_im.set_clim(-vmax, vmax)


def _remove_cs(cs):
    """Elimina un conjunto de curvas de contorno (QuadContourSet) previamente
    dibujado. A diferencia de quiver/imshow/scatter, los objetos que devuelve
    `ax.contour(...)` no tienen un método `set_data` para actualizarse in situ,
    así que hay que borrarlos y volver a crearlos en cada actualización. Se
    prueba `cs.remove()` (API de matplotlib >= 3.8) y, si no existe, se cae al
    método antiguo de recorrer y borrar cada `collection` (versiones previas)."""
    if cs is None:
        return
    try:
        cs.remove()
    except AttributeError:
        for coll in cs.collections:
            coll.remove()


def update_equip_panel():
    """
    Redibuja las curvas equipotenciales (V = constante) en la malla
    intermedia. Se separan explícitamente los niveles positivos y
    negativos de V en dos llamadas a `ax.contour` con estilos distintos
    (línea sólida roja para V>0, línea punteada crema para V<0) para que
    se distinga a simple vista el lado de la carga positiva del lado de
    la carga negativa del dipolo -- con un solo colormap continuo esa
    distinción de signo sería mucho menos evidente.
    """
    _remove_cs(equip_artist["pos"])
    _remove_cs(equip_artist["neg"])
    equip_artist["pos"] = None
    equip_artist["neg"] = None

    Vgrid = potential(X_eq, Y_eq)
    vmax = percentile_abs(Vgrid, 94)
    levels = np.linspace(-0.85 * vmax, 0.85 * vmax, 13)
    levels = levels[np.abs(levels) > 1e-30]
    pos_levels = levels[levels > 0]
    neg_levels = levels[levels < 0]

    # V > 0 (lado de la carga positiva) en rojo brillante
    if pos_levels.size:
        equip_artist["pos"] = ax_equip.contour(
            X_eq, Y_eq, Vgrid, levels=pos_levels, colors=EQUIP_POS, linewidths=1.1,
        )
    # V < 0 (lado de la carga negativa) en crema/dorado claro, bien diferenciado
    if neg_levels.size:
        equip_artist["neg"] = ax_equip.contour(
            X_eq, Y_eq, Vgrid, levels=neg_levels, colors=EQUIP_NEG, linewidths=1.1,
            linestyles="dashed",
        )


def update_divergence_panel():
    """
    Este es el panel que responde directamente a la pregunta de la guía:
    "Discuss why ∇·E vanishes in charge-free regions" (sección 3.3).

    En vez de usar la expresión analítica de ∇·E (que sabemos que es 0
    en el vacío por la propia ley de Gauss, ∇·E = ρ/ε0 con ρ=0 fuera de
    las cargas), aquí se calcula la divergencia de forma puramente
    NUMÉRICA con diferencias finitas centradas, exactamente como se
    haría con datos de campo obtenidos de una simulación o medición:

        ∇·E ≈ ∂Ex/∂x + ∂Ey/∂y

    `np.gradient(Ex, dx, axis=1)` deriva Ex respecto a x (columnas de la
    malla) y `np.gradient(Ey, dy, axis=0)` deriva Ey respecto a y (filas),
    ambas con diferencias centradas de paso dx, dy (el espaciado real de
    la malla `xs_hi`, `ys_hi`).

    El resultado NO es exactamente cero en la práctica: el error de
    discretización es proporcional a dx² y dy² (orden del método de
    diferencias centradas), así que se espera ver un mapa con ruido de
    magnitud pequeña, salvo un pico grande justo sobre cada carga
    puntual (donde ∇·E SÍ debe ser distinto de cero, porque ahí está la
    fuente ρ). Esos picos se recortan explícitamente a 0 con la máscara
    `r < 0.45` porque son un artefacto de la carga puntual idealizada
    (una densidad ρ tipo delta de Dirac no se puede representar bien en
    una malla discreta) y dominarían visualmente el resto del mapa, que
    es la parte que realmente interesa mostrar (∇·E ≈ 0 en el vacío).
    """
    Ex, Ey = e_field(X_hi, Y_hi)
    dx = xs_hi[1] - xs_hi[0]
    dy = ys_hi[1] - ys_hi[0]
    dExdx = np.gradient(Ex, dx, axis=1)
    dEydy = np.gradient(Ey, dy, axis=0)
    div = dExdx + dEydy
    for q, cx, cy, _ in get_charges():
        r = np.hypot(X_hi - cx, Y_hi - cy)
        div = np.where(r < 0.45, 0.0, div)
    vmax = percentile_abs(div, 97)
    div_im.set_data(div)
    div_im.set_clim(-vmax, vmax)


def update_static(_=None):
    update_field_panel()
    update_potential_panel()
    update_equip_panel()
    update_divergence_panel()
    draw_charge_markers()
    for cbar, im in ((pot_cbar, pot_im), (div_cbar, div_im)):
        cbar.update_normal(im)
    fig.canvas.draw_idle()


# ----------------------------------------------------------------------
# Animación de partículas de flujo (matplotlib.animation.FuncAnimation)
# ----------------------------------------------------------------------
def animate(_frame):
    """
    Se llama automáticamente ~25 veces por segundo (interval=40 ms, ver
    `FuncAnimation` al final del script). En cada llamada:

    1. Se evalúa el campo eléctrico en la posición actual de cada
       partícula (`e_field` vectorizado sobre las 160 partículas a la vez).
    2. Cada partícula avanza un paso pequeño (`step = 0.05`) en la
       dirección UNITARIA del campo local (Ex/|E|, Ey/|E|) -- es decir,
       todas avanzan a la misma velocidad visual, independientemente de
       si están cerca o lejos de una carga. Esto es una integración de
       Euler explícita muy simple de la ecuación dr/dt = Ê(r); no busca
       precisión física sino dar una sensación fluida de "seguir las
       líneas de campo" en tiempo real.
    3. Las partículas que salen del recuadro visible o que caen
       demasiado cerca de una carga (r < 3·SOFT, donde la dirección del
       campo se vuelve numéricamente inestable) se "reciclan": se les
       asigna una nueva posición aleatoria con `respawn`, de forma que
       el número de partículas en pantalla siempre es constante.

    FuncAnimation espera que la función de callback devuelva una tupla
    con los artistas que cambiaron (aquí solo `flow_scatter`), lo cual
    matplotlib usa para optimizar el redibujado cuando `blit=True`
    (en este script se usa `blit=False`, más simple pero algo menos
    eficiente).
    """
    global particles
    Ex, Ey = e_field(particles[:, 0], particles[:, 1])
    mag = np.hypot(Ex, Ey)
    mag_safe = np.maximum(mag, 1e-30)
    step = 0.05
    particles[:, 0] += step * Ex / mag_safe
    particles[:, 1] += step * Ey / mag_safe

    out = (
        (particles[:, 0] < XMIN) | (particles[:, 0] > XMAX)
        | (particles[:, 1] < YMIN) | (particles[:, 1] > YMAX)
    )
    for q, cx, cy, _ in get_charges():
        r = np.hypot(particles[:, 0] - cx, particles[:, 1] - cy)
        out |= r < SOFT * 3
    n_out = int(np.sum(out))
    if n_out:
        particles[out] = respawn(n_out)

    flow_scatter.set_offsets(particles)
    return (flow_scatter,)


# ----------------------------------------------------------------------
# Controles (sliders y botones de radio) — variación de parámetros
# ----------------------------------------------------------------------
# Esta sección implementa el punto 4 "Parameter variation" de la guía:
# permite explorar cómo cambian E, V, las equipotenciales y ∇·E al variar
# la magnitud de la carga (q) y la separación del dipolo (d), y al cambiar
# entre las configuraciones (a) carga puntual y (b) dipolo.
ax_q = fig.add_axes([0.10, 0.15, 0.35, 0.03])
ax_q.set_facecolor(LINE)
slider_q = Slider(ax_q, "q (nC)", 0.2, 5.0, valinit=state["q_nC"], color=AMBER)
slider_q.label.set_color(INK_DIM)
slider_q.valtext.set_color(INK)

ax_d = fig.add_axes([0.10, 0.09, 0.35, 0.03])
ax_d.set_facecolor(LINE)
slider_d = Slider(ax_d, "d (m)", 0.5, 4.0, valinit=state["d_m"], color=TEAL)
slider_d.label.set_color(INK_DIM)
slider_d.valtext.set_color(INK)

ax_radio = fig.add_axes([0.58, 0.06, 0.20, 0.13], facecolor=PANEL_BG)
radio = RadioButtons(ax_radio, ("Dipolo", "Carga puntual"), active=0, activecolor=CORAL)
try:
    radio.set_radio_props({"edgecolor": INK_DIM, "facecolor": PANEL_BG})
except AttributeError:
    for circ in radio.circles:  # matplotlib < 3.7 fallback
        circ.set_edgecolor(INK_DIM)
        circ.set_facecolor(PANEL_BG)
for lbl in radio.labels:
    lbl.set_color(INK)
    lbl.set_fontsize(10)


def on_q_change(val):
    """Callback del slider de carga: actualiza el estado global y vuelve a
    calcular/dibujar los 4 paneles estáticos (no toca la animación de
    partículas, que sigue corriendo con `e_field` leyendo el nuevo `state`
    automáticamente en el siguiente frame)."""
    state["q_nC"] = val
    update_static()


def on_d_change(val):
    """Callback del slider de separación d (solo tiene efecto visible en
    modo dipolo, aunque el valor se sigue guardando en modo carga puntual
    por si el usuario vuelve a cambiar de modo)."""
    state["d_m"] = val
    update_static()


def on_mode_change(label):
    """Callback de los botones de radio: cambia entre 'Dipolo' y
    'Carga puntual'. Además de recalcular los paneles, oculta el slider
    de separación d cuando no aplica (modo carga puntual), para no
    confundir al usuario con un control sin efecto."""
    state["mode"] = "dipole" if label == "Dipolo" else "point"
    ax_d.set_visible(state["mode"] == "dipole")
    update_static()


slider_q.on_changed(on_q_change)
slider_d.on_changed(on_d_change)
radio.on_clicked(on_mode_change)


def on_move(event):
    """
    Se conecta al evento 'motion_notify_event' de matplotlib: se dispara
    cada vez que el mouse se mueve dentro de la figura. Si el mouse está
    sobre el panel de campo (ax_field), se evalúan E y V exactamente en
    esas coordenadas de datos (event.xdata, event.ydata ya vienen
    convertidas de píxeles de pantalla a metros del sistema de ejes) y se
    muestra el resultado numérico en el texto superior. Es una forma
    sencilla de "sonda de campo" virtual, útil para verificar valores
    puntuales sin tener que leerlos de la escala de color.
    """
    if event.inaxes is not ax_field or event.xdata is None:
        return
    x, y = event.xdata, event.ydata
    Ex, Ey = e_field(np.array([x]), np.array([y]))
    mag = float(np.hypot(Ex[0], Ey[0]))
    v = float(potential(np.array([x]), np.array([y]))[0])
    readout_text.set_text(
        f"en ({x:.2f}, {y:.2f}) m   |E| = {mag:.2e} N/C   V = {v:.2e} V"
    )
    fig.canvas.draw_idle()


def on_leave(event):
    """Se conecta a 'figure_leave_event': cuando el mouse sale de la
    ventana de la figura, se restaura el texto de instrucción original."""
    readout_text.set_text("Mueve el mouse sobre el panel de campo para leer E y V.")
    fig.canvas.draw_idle()


fig.canvas.mpl_connect("motion_notify_event", on_move)
fig.canvas.mpl_connect("figure_leave_event", on_leave)

fig.text(
    0.06, 0.03,
    "Modelo con ablandamiento (softening = 0.05) para evitar divergencias sobre la carga.\n"
    "Dominio: x ∈ [−4.5, 4.5] m, y ∈ [−1.8, 1.8] m.  Fuera de las cargas, ∇·E ≈ 0 (Ley de Gauss).",
    color=INK_DIM, fontsize=8, family="monospace",
)

# ----------------------------------------------------------------------
# Inicialización y arranque
# ----------------------------------------------------------------------
update_static()  # dibuja los 4 paneles con los valores iniciales de `state`

# `anim` debe guardarse en una variable a nivel de módulo: si se creara la
# animación dentro de una función y no se conservara ninguna referencia a
# ella, Python la recolectaría como basura (garbage collection) y la
# animación se detendría silenciosamente apenas terminara esa función.
# interval=40 ms ≈ 25 fps; cache_frame_data=False evita que matplotlib
# intente guardar en memoria todos los frames de una animación "infinita".
anim = FuncAnimation(fig, animate, interval=40, blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()