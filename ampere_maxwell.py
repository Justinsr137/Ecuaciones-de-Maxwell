
"""
Ley de Ampère-Maxwell y propagación de ondas electromagnéticas — FDTD 1D
==========================================================================
Este script corresponde a la sección 6 ("Ampère–Maxwell law",
∇×B = μ₀J + μ₀ε₀∂E/∂t) de la actividad "Computational electrodynamics
training guide", en su configuración (c) "propagating electromagnetic
pulse". También sirve de ejemplo para la sección 7 "Full-system
integration", ya que la onda que aquí se propaga es precisamente la
solución de la ecuación de onda ∇²E - μ₀ε₀ ∂²E/∂t² = 0 que resulta de
combinar la ley de Faraday y la de Ampère-Maxwell, y su velocidad de
propagación en el vacío debe coincidir con c = 1/√(μ₀ε₀).

A diferencia de los scripts de Gauss (donde el campo se evalúa con una
fórmula analítica cerrada), aquí NO hay fórmula cerrada: el campo se
obtiene resolviendo las ecuaciones de Maxwell dependientes del tiempo con
el método FDTD (Finite-Difference Time-Domain, diferencias finitas en el
dominio del tiempo, algoritmo de Yee), que es el método numérico estándar
en electrodinámica computacional para propagación de ondas.

Ecuaciones que se resuelven (onda plana 1D, polarizada con E en ẑ y
H en ŷ, propagándose en x̂ — el caso más simple de onda TEM):

    Ley de Faraday          ∂Hy/∂t = -(1/μ0) · ∂Ez/∂x
    Ley de Ampère-Maxwell   ∂Ez/∂t =  (1/ε)  · ∂Hy/∂x

Es decir, el script en realidad integra el sistema ACOPLADO de las dos
ecuaciones rotacionales de Maxwell (Faraday da la actualización de H a
partir de E, Ampère-Maxwell da la actualización de E a partir de H); se
tituló pensando en la ley de Ampère-Maxwell porque es la que aporta el
término de "corriente de desplazamiento" (ε·∂E/∂t) que hace posible que
el campo se propague como onda sin necesidad de corriente de conducción
J, que es el resultado central que pide analizar la sección 6.3 de la
guía.

Elementos físicos incluidos:
- Malla de Yee (Ez y Hy definidos en puntos intercalados/"staggered").
- Esquema de diferencias finitas de 4º orden en el interior del dominio
  (más preciso, menos dispersión numérica, que el esquema clásico de
  2º orden de Yee) con un esquema de 2º orden simple cerca de los bordes.
- Fuente "suave" (soft source) en el extremo izquierdo: una onda senoidal
  con una envolvente tipo tanh que se enciende gradualmente, para no
  inyectar componentes de alta frecuencia espurias por un arranque abrupto.
- Condición de frontera absorbente de Mur (1er orden) en el extremo
  derecho: una fórmula que aproxima una "frontera abierta" para que la
  onda salga del dominio simulado sin reflejarse artificialmente.
- Una interfaz entre dos medios (vacío y un medio con permitividad
  relativa εr ajustable), para observar transmisión/reflexión parcial en
  el cambio de medio (similar a una onda de luz entrando a un vidrio).
- Vector de Poynting S = E×H, visualizado con flechas, que indica la
  dirección y magnitud del flujo de energía electromagnética.

Controles interactivos:
- Slider **f** — frecuencia de la fuente.
- Slider **E₀** — amplitud del campo eléctrico inyectado.
- Slider **εr (medio)** — permitividad relativa del segundo medio
  (x > interfaz), que cambia la velocidad de fase v = c0/√εr en esa
  región y el coeficiente de reflexión en la interfaz.
- Checkboxes para mostrar/ocultar: campo E, campo H, vector de Poynting,
  interfaz de medio, rotación automática de la cámara.
- Botones **Pausa** y **Reiniciar**.

Requisitos:  pip install numpy matplotlib
Ejecutar:    python ampere_maxwell.py
(usa el backend TkAgg, fijado explícitamente al inicio del script; si no
está disponible, instala `python3-tk` o cambia a otro backend interactivo)
"""

import numpy as np

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# COLORES

BG = "#050303"
PANEL_BG = "#0A0404"
LINE_PANEL = "#5C1414"

TEXT = "#F2E9E7"
TEXT_DIM = "#B08383"

# Campo magnético H (rojo, como el B del embobinado)
B_COLOR = "#FF1E1E"

# Vector de Poynting
S_COLOR = "#39FF14"

# Campo eléctrico E (azul)
E_COLOR = "#55B7FF"

# Color del segundo medio: gris translúcido
MEDIUM_COLOR = "#9A9A9A"
MEDIUM_ALPHA = 0.16

TITLE_COLOR = "#FF453A"

#PARÁMETROS DEL SISTEMA

Nx = 420          # número de puntos de la malla de Ez (campo eléctrico)
Lx = 22.0         # longitud del dominio simulado [unidades naturales]
dx = Lx / Nx      # paso espacial

eps0 = 1.0        # permitividad del vacío
mu0 = 1.0         # permeabilidad del vacío
                  # (se usan unidades "naturales" eps0 = mu0 = 1, no
                  # unidades SI: así c0 = 1 exactamente, lo que simplifica
                  # la lectura de los resultados en términos relativos;
                  # la física de la propagación no cambia, solo la escala)
c0 = 1.0 / np.sqrt(eps0 * mu0)   # velocidad de la luz en el vacío (=1 aquí)


S_courant = 0.5   # número de Courant S = c0·dt/dx (debe ser <= 1 en 1D
                   # para que el esquema FDTD explícito sea estable; 0.5
                   # da margen extra de estabilidad frente al esquema de
                   # 4º orden, que es algo más exigente que el de 2º orden)
dt = S_courant * dx / c0   # paso temporal, fijado por la condición CFL

# Malla de Yee ("staggered grid"): Ez se define en los puntos enteros
# x_E = 0, dx, 2dx, ..., mientras que Hy se define en los puntos medios
# x_H = dx/2, 3dx/2, ... Este entrelazado (una malla desplazada media
# celda respecto a la otra) es la base del algoritmo de Yee: permite
# aproximar las derivadas espaciales con diferencias centradas de 2º
# orden de precisión sin necesidad de "duplicar" puntos, y es la forma
# estándar de discretizar las ecuaciones de Maxwell dependientes del
# tiempo en FDTD.
x_E = np.linspace(0.0, Lx, Nx)
x_H = x_E[:-1] + dx / 2.0

interface_index = Nx // 2   # índice de x_E donde comienza el segundo medio


# SOLUCIÓN  DEL SISTEMA

class WaveFDTD:
    """
    Encapsula el estado y la evolución temporal del campo electromagnético
    1D mediante el algoritmo de Yee (FDTD). Cada llamada a `step()` avanza
    la simulación un paso de tiempo dt, actualizando primero Hy (con la
    ley de Faraday) y luego Ez (con la ley de Ampère-Maxwell), que es el
    orden característico del "leapfrog" de Yee: los dos campos se evalúan
    en instantes de tiempo intercalados (Hy en t, t+dt, t+2dt, ... y Ez en
    t+dt/2, t+3dt/2, ...), aunque aquí se maneja de forma simplificada
    actualizando ambos dentro de la misma llamada a `step`.
    """

    def __init__(self):

        self.f = 0.15        # frecuencia de la fuente senoidal
        self.E0 = 1.6        # amplitud del campo eléctrico inyectado
        self.epsr = 1.0      # permitividad relativa del 2º medio (x grande)

        self.Ez = np.zeros(Nx)      # campo eléctrico en la malla x_E
        self.Hy = np.zeros(Nx - 1)  # campo magnético en la malla x_H

        self.t = 0.0

    def eps_profile(self):
        """
        Construye el perfil espacial de permitividad ε(x): vale ε0=1 en
        la primera mitad del dominio (x < x_iface, "vacío") y ε0·εr en la
        segunda mitad (x >= x_iface, el "medio" cuya permitividad relativa
        se controla con el slider). Este perfil es lo que introduce la
        interfaz dieléctrica: al cambiar el medio cambia tanto la
        velocidad de fase local (v = c0/√εr) como la impedancia, lo que
        produce reflexión parcial de la onda en la interfaz.
        """
        eps = np.ones(Nx)

        eps[interface_index:] = self.epsr

        return eps * eps0

    def step(self):
        """
        Avanza la simulación un paso de tiempo dt. Estructura del método
        (el orden de las 4 partes importa: es el ciclo "leapfrog" de Yee):

        1) ACTUALIZAR H a partir del E actual (ley de Faraday discretizada:
           ∂Hy/∂t = -(1/μ0)·∂Ez/∂x  →  Hy_nuevo = Hy_viejo + (dt/μ0)·dEz/dx).
        2) ACTUALIZAR E a partir del H recién actualizado (ley de
           Ampère-Maxwell discretizada:
           ∂Ez/∂t = (1/ε)·∂Hy/∂x  →  Ez_nuevo = Ez_viejo + (dt/ε)·dHy/dx).
        3) INYECTAR la fuente en el extremo izquierdo (sobrescribe Ez[0]).
        4) APLICAR la condición de frontera absorbente en el extremo
           derecho (sobrescribe Ez[-1]).

        Las derivadas espaciales (dEz/dx, dHy/dx) se calculan con un
        esquema de 4º orden de precisión en el interior del dominio (más
        preciso, con menor "dispersión numérica" -- el error por el cual
        distintas frecuencias viajarían a distinta velocidad solo por la
        discretización -- que el esquema estándar de 2º orden de Yee), y
        con diferencias simples de 2º orden en los dos puntos más
        cercanos a cada borde, donde no hay suficientes vecinos para
        aplicar la fórmula de 4º orden.
        """
        eps = self.eps_profile()

        Ez_old = self.Ez.copy()

        # --- (1) Actualización de H (ley de Faraday) ---------------------
        # Derivada dEz/dx en los puntos de la malla de H (x_H), con
        # diferencias centradas de 2º orden como valor por defecto:
        dEdx = (self.Ez[1:] - self.Ez[:-1]) / dx

        # Se reemplaza esa derivada por una fórmula de 4º orden en los
        # puntos interiores donde hay suficientes vecinos a ambos lados
        # (stencil de 4 puntos: i-1, i, i+1, i+2), una versión compacta
        # de la fórmula de diferencias finitas centradas de orden 4:
        i4 = np.arange(1, Nx - 2)

        dEdx[i4] = (
            9.0 * (self.Ez[i4 + 1] - self.Ez[i4])
            -
            (self.Ez[i4 + 2] - self.Ez[i4 - 1])
        ) / (8.0 * dx)

        self.Hy += (dt / mu0) * dEdx

        # --- (2) Actualización de E (ley de Ampère-Maxwell) --------------
        # Misma idea que arriba pero para dHy/dx, evaluada en los puntos
        # interiores de Ez (Ez[1:-1]; los extremos Ez[0] y Ez[-1] se fijan
        # aparte, por la fuente y la frontera absorbente):
        dHdx = (self.Hy[1:] - self.Hy[:-1]) / dx  # cubre Ez[1:-1]

        i4e = np.arange(2, Nx - 2)
        pos = i4e - 1

        dHdx[pos] = (
            9.0 * (self.Hy[i4e] - self.Hy[i4e - 1])
            -
            (self.Hy[i4e + 1] - self.Hy[i4e - 2])
        ) / (8.0 * dx)

        # Nótese que aquí se divide por ε(x) (posiblemente distinto de
        # eps0 después de la interfaz), no por eps0: así la velocidad de
        # la onda se reduce automáticamente en el medio con εr > 1.
        self.Ez[1:-1] += (dt / eps[1:-1]) * dHdx

        self.t += dt

        # --- (3) Fuente "suave" en el extremo izquierdo -------------------
        # Se inyecta una onda senoidal de frecuencia f, pero multiplicada
        # por una envolvente tanh que crece de 0 a 1: al t=0 la fuente
        # empieza apagada y se enciende gradualmente en unos pocos
        # periodos. Sin esta envolvente, un salto abrupto de 0 a la
        # amplitud máxima en un solo paso de tiempo introduciría
        # componentes de muy alta frecuencia (un "clic" numérico) que se
        # propagarían como ruido espurio por todo el dominio.
        envelope = np.tanh(
            self.t * max(self.f, 1e-6) * 4.0
        )

        self.Ez[0] = (
            self.E0
            * envelope
            * np.sin(2 * np.pi * self.f * self.t)
        )

        # --- (4) Frontera absorbente de Mur (1er orden) en x = Lx ---------
        # Sin esta condición, la onda se reflejaría por completo al llegar
        # al borde derecho del dominio (como si fuera una pared), lo cual
        # no representa un espacio abierto/infinito. La condición de Mur
        # de 1er orden aproxima una ecuación de onda unidireccional
        # (∂E/∂t + c·∂E/∂x = 0, que solo admite ondas viajando hacia +x)
        # evaluada en el borde, dejando "salir" la onda con una reflexión
        # residual pequeña (del orden del error de discretización, no
        # cero exacto). `c_local` usa la velocidad de fase del medio en el
        # que se encuentra el borde derecho (que puede ser el medio con εr).
        c_local = c0 / np.sqrt(self.epsr)

        coeff = (
            (c_local * dt - dx)
            /
            (c_local * dt + dx)
        )

        self.Ez[-1] = (
            Ez_old[-2]
            +
            coeff * (self.Ez[-2] - Ez_old[-1])
        )

        return self.Ez, self.Hy

    def reset(self):
        """Apaga los campos y reinicia el reloj de la simulación (se llama
        desde el botón 'Reiniciar' de la interfaz)."""
        self.Ez[:] = 0.0
        self.Hy[:] = 0.0
        self.t = 0.0


state = WaveFDTD()

# Dibujar una flecha en cada uno de los ~420 puntos de la malla sería
# ilegible; en cambio se eligen subconjuntos de índices, espaciados
# uniformemente, para las flechas de E, de H y del vector de Poynting.
n_arrows = 26

arrow_idx_E = np.linspace(
    2,
    Nx - 3,
    n_arrows
).astype(int)

arrow_idx_H = np.clip(
    arrow_idx_E,
    0,
    Nx - 2
)

n_poynting = 12

poynting_idx = np.linspace(
    3,
    Nx - 4,
    n_poynting
).astype(int)

# GRAFICA
# La escena 3D no representa un espacio 3D físico real: el eje x es la
# dirección de propagación real, pero los ejes y y z se usan simplemente
# para "levantar" y separar visualmente las curvas de Ez (graficada en el
# plano y) y de Hy (graficada en el plano z), ya que en la polarización
# elegida (TEM con E en ẑ, H en ŷ) ambos campos son perpendiculares entre
# sí y a la dirección de propagación -- exactamente lo que se busca
# mostrar de una onda electromagnética.
fig = plt.figure(
    figsize=(24, 15.5),
    facecolor=BG
)

ax = fig.add_axes(
    [0.03, 0.30, 0.60, 0.66],
    projection="3d",
    facecolor=BG
)

ax.set_axis_off()

field_max = 3.0   # límite de los ejes y/z (rango visual de amplitud de campo)

ax.set_xlim(0.0, Lx)
ax.set_ylim(-field_max, field_max)
ax.set_zlim(-field_max, field_max)

ax.set_box_aspect(
    [Lx, field_max * 2, field_max * 2]
)

ax.view_init(
    elev=16,
    azim=-58
)


# Eje de propagación: una simple línea punteada a lo largo de x, de
# referencia visual (no tiene ningún papel físico en el cálculo).
propagation_axis, = ax.plot(
    [0.0, Lx],
    [0.0, 0.0],
    [0.0, 0.0],
    color=TEXT_DIM,
    linewidth=1.0,
    linestyle="--",
    alpha=0.6
)

x_iface = x_E[interface_index]


def box_faces(x0, x1, y0, y1, z0, z1):
    """Devuelve las 6 caras (cada una como lista de 4 vértices) de una
    caja rectangular alineada a los ejes, en el formato que espera
    Poly3DCollection. Se usa para dibujar el bloque semitransparente que
    representa visualmente el segundo medio (x > x_iface)."""

    return [
        [[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0]],  # abajo
        [[x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]],  # arriba
        [[x0, y0, z0], [x0, y1, z0], [x0, y1, z1], [x0, y0, z1]],  # cara x0
        [[x1, y0, z0], [x1, y1, z0], [x1, y1, z1], [x1, y0, z1]],  # cara x1
        [[x0, y0, z0], [x1, y0, z0], [x1, y0, z1], [x0, y0, z1]],  # frente
        [[x0, y1, z0], [x1, y1, z0], [x1, y1, z1], [x0, y1, z1]],  # atrás
    ]


# El bloque semitransparente cubre todo el rango visible de x a partir de
# la interfaz, ilustrando dónde "empieza" el medio con permitividad εr.
medium_faces = box_faces(
    x_iface, Lx,
    -field_max, field_max,
    -field_max, field_max
)

interface_collection = Poly3DCollection(
    medium_faces,
    facecolor=MEDIUM_COLOR,
    edgecolor=MEDIUM_COLOR,
    alpha=MEDIUM_ALPHA,
    linewidth=0.4
)

ax.add_collection3d(
    interface_collection
)


#CAMPOS
# `line_E` grafica Ez(x) sobre el plano y (Ez a lo largo de x, con y=0
# fijo), y `line_H` grafica Hy(x) sobre el plano z: así, aunque ambas
# curvas comparten el mismo eje x de propagación, se distinguen
# visualmente como dos ondas perpendiculares entre sí, tal como son E y H
# físicamente en una onda TEM. Son artistas persistentes: en cada frame de
# la animación se actualizan con `set_data_3d`, no se recrean.

line_E, = ax.plot(
    x_E,
    state.Ez,
    np.zeros_like(x_E),
    color=E_COLOR,
    linewidth=2.6
)

line_H, = ax.plot(
    x_H,
    np.zeros_like(x_H),
    state.Hy,
    color=B_COLOR,
    linewidth=2.6
)


# Las flechas (quiver 3D) sí hay que recrearlas en cada frame (no se
# pueden "mover" in situ como una línea), así que se guardan en estas
# listas para poder borrarlas antes de dibujar el siguiente frame.
E_arrow_objects = []
H_arrow_objects = []
S_arrow_objects = []


def clear_arrows(objects):
    """Elimina del eje 3D todas las flechas guardadas en `objects` (una
    lista de artistas `Quiver`) y vacía la lista. Se llama al inicio de
    cada frame de animación, antes de volver a dibujar las flechas con
    los valores de campo actualizados."""
    for obj in objects:

        try:
            obj.remove()
        except Exception:
            pass

    objects.clear()



ax.set_title(
    "ONDA ELECTROMAGNÉTICA — LEY DE AMPÈRE-MAXWELL ",
    color=TITLE_COLOR,
    fontsize=20,
    y=1.03,
    pad=18
)



# Panel de datos

info_ax = fig.add_axes(
    [0.69, 0.035, 0.28, 0.335],
    facecolor=PANEL_BG
)

info_ax.set_xticks([])
info_ax.set_yticks([])

for spine in info_ax.spines.values():
    spine.set_color(LINE_PANEL)

info = info_ax.text(
    0.045,
    0.955,
    "",
    transform=info_ax.transAxes,
    color=TEXT,
    fontsize=14.5,
    family="monospace",
    va="top",
    ha="left",
    linespacing=1.65
)


slider_axes = {}

slider_positions = {
    "f": 0.235,
    "E0": 0.195,
    "epsr": 0.155
}

for name, y in slider_positions.items():

    slider_axes[name] = fig.add_axes(
        [0.08, y, 0.50, 0.022],
        facecolor=LINE_PANEL
    )


s_f = Slider(
    slider_axes["f"],
    "f [1/t]",
    0.05,
    0.40,
    valinit=state.f,
    valstep=0.01,
    color=E_COLOR
)

# E₀: amplitud de la fuente en el extremo izquierdo (ver `WaveFDTD.step`).
s_E0 = Slider(
    slider_axes["E0"],
    "E₀ [amp.]",
    0.3,
    2.8,
    valinit=state.E0,
    valstep=0.1,
    color=E_COLOR
)

s_epsr = Slider(
    slider_axes["epsr"],
    "εr (medio)",
    1.0,
    4.0,
    valinit=state.epsr,
    valstep=0.1,
    color=MEDIUM_COLOR
)
# εr controla la permitividad relativa del medio a la derecha de la
# interfaz (ver `eps_profile`): al aumentarla, la onda viaja más lento
# en esa región (v = c0/√εr) y aparece reflexión parcial en la interfaz
# (ver el cálculo de `r_interfaz` dentro de `update`, más abajo).


for slider in (s_f, s_E0, s_epsr):

    slider.label.set_color(TEXT)
    slider.valtext.set_color(TEXT)
    slider.label.set_fontsize(11)


#
# Checkboxes para mostrar/ocultar cada capa visual. El bloque try/except
# existe por COMPATIBILIDAD entre versiones de Matplotlib: las versiones
# recientes de CheckButtons aceptan los parámetros `label_props`,
# `frame_props`, `check_props` para personalizar colores directamente en
# el constructor; versiones más antiguas no los reconocen (lanzan
# TypeError) y hay que acceder a los atributos `labels`/`rectangles`/
# `lines` del objeto ya creado para colorearlos manualmente. Así el
# script funciona igual de bien en ambos casos sin tener que fijar una
# versión exacta de Matplotlib como requisito.
check_ax = fig.add_axes(
    [0.69, 0.415, 0.28, 0.535],
    facecolor=PANEL_BG
)

check_labels = [
    "Campo eléctrico E",
    "Campo magnético H",
    "Vector de Poynting (E×H)",
    "Interfaz de medio (εr)",
    "Rotación automática"
]

check_actives = [True, True, True, True, False]
n_checks = len(check_labels)

try:

    check = CheckButtons(
        check_ax,
        check_labels,
        actives=check_actives,
        label_props={
            "color": [TEXT] * n_checks,
            "fontsize": [12.5] * n_checks
        },
        frame_props={
            "facecolor": [PANEL_BG] * n_checks,
            "edgecolor": [TEXT_DIM] * n_checks,
            "linewidth": [1.3] * n_checks
        },
        check_props={
            "facecolor": [S_COLOR] * n_checks,
            "linewidth": [2.4] * n_checks
        }
    )

except TypeError:


    check = CheckButtons(
        check_ax,
        check_labels,
        check_actives
    )

    for label in check.labels:

        label.set_color(TEXT)
        label.set_fontsize(12.5)

    if hasattr(check, "rectangles"):

        for rect in check.rectangles:

            rect.set_facecolor(PANEL_BG)
            rect.set_edgecolor(TEXT_DIM)
            rect.set_linewidth(1.3)

    if hasattr(check, "lines"):

        for l1, l2 in check.lines:

            l1.set_color(S_COLOR)
            l2.set_color(S_COLOR)
            l1.set_linewidth(2.4)
            l2.set_linewidth(2.4)

for spine in check_ax.spines.values():
    spine.set_color(LINE_PANEL)



pause_ax = fig.add_axes(
    [0.08, 0.015, 0.14, 0.04]
)

reset_ax = fig.add_axes(
    [0.24, 0.015, 0.14, 0.04]
)

pause_button = Button(
    pause_ax,
    "Pausa",
    color=PANEL_BG,
    hovercolor=LINE_PANEL
)

reset_button = Button(
    reset_ax,
    "Reiniciar",
    color=PANEL_BG,
    hovercolor=LINE_PANEL
)

pause_button.label.set_color(TEXT)
reset_button.label.set_color(TEXT)

running = [True]   # bandera mutable: True = la simulación avanza en cada
                    # frame; False = update() sigue redibujando la
                    # interfaz pero deja de llamar a state.step()


def pause_callback(event):
    """Alterna el estado de pausa y actualiza el texto del botón para
    reflejarlo ('Pausa' <-> 'Reanudar')."""
    running[0] = not running[0]

    if running[0]:
        pause_button.label.set_text("Pausa")
    else:
        pause_button.label.set_text("Reanudar")


def reset_callback(event):
    """Reinicia la simulación (campos a cero, t=0) y limpia todas las
    flechas dibujadas, para volver a un estado limpio sin cerrar la
    ventana."""
    state.reset()

    clear_arrows(E_arrow_objects)
    clear_arrows(H_arrow_objects)
    clear_arrows(S_arrow_objects)

    fig.canvas.draw_idle()


pause_button.on_clicked(pause_callback)
reset_button.on_clicked(reset_callback)




def parameter_callback(value):
    """Callback común para los tres sliders físicos: simplemente vuelca
    sus valores actuales al objeto `state`, que es de donde los lee
    `WaveFDTD.step()` en el siguiente paso de la simulación. Nótese que
    cambiar εr o f NO reinicia la simulación: el cambio se aplica de
    forma continua sobre los campos ya existentes, lo que puede generar
    transitorios interesantes de observar (por ejemplo, cambiar εr
    mientras el pulso ya está viajando)."""
    state.f = s_f.val
    state.E0 = s_E0.val
    state.epsr = s_epsr.val


for slider in (s_f, s_E0, s_epsr):

    slider.on_changed(parameter_callback)



def check_callback(label):
    """Callback de las casillas de verificación. Las capas 'Vector de
    Poynting' y 'Rotación automática' (índices 2 y 4) no se controlan
    aquí porque no corresponden a un artista fijo: su visibilidad se
    decide cada frame dentro de `update()`, leyendo `check.get_status()`
    directamente en el momento de dibujar."""
    status = check.get_status()

    line_E.set_visible(status[0])
    line_H.set_visible(status[1])

    interface_collection.set_visible(status[3])

    fig.canvas.draw_idle()


check.on_clicked(check_callback)




def draw_field_arrows(
    x_positions,
    values,
    axis,
    color,
    holder
):
    """
    Dibuja una flecha por cada posición en `x_positions`, partiendo del
    eje de propagación (y=0, z=0) y con longitud/dirección igual al valor
    del campo en ese punto (`values`). `axis` indica sobre qué eje crece
    la flecha: "y" para el campo eléctrico (Ez se grafica en el plano y,
    ver `line_E`) o "z" para el campo magnético (Hy se grafica en el
    plano z, ver `line_H`) — así las flechas quedan alineadas con la
    curva correspondiente. Los objetos `Quiver` creados se van agregando
    a `holder` (una de las listas E_arrow_objects/H_arrow_objects) para
    poder borrarlos en el siguiente frame con `clear_arrows`.
    """
    for xp, val in zip(x_positions, values):

        if axis == "y":
            dvec = (0.0, val, 0.0)
        else:
            dvec = (0.0, 0.0, val)

        q = ax.quiver(
            xp,
            0.0,
            0.0,
            dvec[0],
            dvec[1],
            dvec[2],
            color=color,
            linewidth=1.6,
            arrow_length_ratio=0.35
        )

        holder.append(q)


def draw_poynting_arrows(idx, Ez, Hy, holder):
    """
    Dibuja flechas a lo largo del eje x que representan el vector de
    Poynting S = E×H, la magnitud que describe la dirección y densidad de
    flujo de energía electromagnética. Como Ez vive en la malla x_E y Hy
    en la malla x_H (desplazadas media celda, ver la sección de la malla
    de Yee), primero se interpola Hy al punto x_E[i] promediando sus dos
    vecinos más cercanos (`Hy_interp`).

    Para la polarización elegida (E en ẑ, H en ŷ), el producto cruz
    E×H = (Ez ẑ)×(Hy ŷ) = -Ez·Hy x̂ (usando ẑ×ŷ = -x̂), de ahí el signo
    menos en `Sx = -Ez[i] * Hy_interp`. El resultado se recorta
    (`np.clip`) para que la longitud de la flecha no se salga del rango
    visual `field_max`, y se dibuja únicamente sobre el eje x (dirección
    de propagación), que es la dirección física real de S para esta onda.
    """
    for i in idx:

        Hy_interp = 0.5 * (
            Hy[max(i - 1, 0)]
            +
            Hy[min(i, len(Hy) - 1)]
        )

        Sx = -Ez[i] * Hy_interp

        q = ax.quiver(
            x_E[i],
            0.0,
            0.0,
            np.clip(Sx * 2.5, -field_max, field_max),
            0.0,
            0.0,
            color=S_COLOR,
            linewidth=1.4,
            arrow_length_ratio=0.4
        )

        holder.append(q)


# ANIMACIÓN

# Se ejecutan varios pasos de FDTD por cada frame dibujado. El paso de
# tiempo dt de la simulación es muy pequeño (fijado por la condición
# CFL/Courant para estabilidad numérica, ver `dt` arriba), así que
# avanzar un solo `dt` por frame haría que la onda se viera casi
# congelada a 25 fps; ejecutar 13 sub-pasos de física por cada frame de
# dibujo permite ver una propagación fluida sin sacrificar la resolución
# temporal necesaria para la estabilidad del esquema numérico.
substeps_per_frame = 13


def update(frame):
    """Callback de animación (~25 fps). Si la simulación no está en
    pausa, avanza `substeps_per_frame` pasos de FDTD; si está en pausa,
    simplemente vuelve a leer el último estado calculado (sin avanzar el
    tiempo), para que los checkboxes y el panel de datos sigan
    respondiendo aunque la física esté detenida."""
    if running[0]:

        for _ in range(substeps_per_frame):
            Ez, Hy = state.step()

    else:

        Ez, Hy = state.Ez, state.Hy

    status = check.get_status()

    # Actualiza las curvas persistentes de E y H con los nuevos valores.
    line_E.set_data_3d(
        x_E,
        Ez,
        np.zeros_like(x_E)
    )

    line_H.set_data_3d(
        x_H,
        np.zeros_like(x_H),
        Hy
    )

    # Las flechas SIEMPRE se limpian y, si la capa correspondiente está
    # activa en los checkboxes, se vuelven a dibujar desde cero con los
    # valores de campo del frame actual.
    clear_arrows(E_arrow_objects)
    clear_arrows(H_arrow_objects)
    clear_arrows(S_arrow_objects)

    if status[0]:

        draw_field_arrows(
            x_E[arrow_idx_E],
            Ez[arrow_idx_E],
            "y",
            E_COLOR,
            E_arrow_objects
        )

    if status[1]:

        draw_field_arrows(
            x_H[arrow_idx_H],
            Hy[arrow_idx_H],
            "z",
            B_COLOR,
            H_arrow_objects
        )

    if status[2]:

        draw_poynting_arrows(
            poynting_idx,
            Ez,
            Hy,
            S_arrow_objects
        )

    # --- Panel de datos: cantidades derivadas de los parámetros actuales ---
    wavelength_vac = c0 / max(state.f, 1e-6)   # λ = c/f en el vacío
    v_medio = c0 / np.sqrt(state.epsr)          # velocidad de fase en el 2º medio
    n_medio = np.sqrt(state.epsr)               # índice de refracción n = √εr

    # Coeficiente de reflexión de amplitud en incidencia normal entre dos
    # medios nn (aquí, vacío n=1 al segundo medio de índice n_medio),
    # análogo a la fórmula de Fresnel r = (n1-n2)/(n1+n2):
    r_interfaz = (1.0 - n_medio) / (1.0 + n_medio)

    info.set_text(
        f"t         = {state.t:.4f}\n"
        f"λ (vacío) = {wavelength_vac:.4f}\n"
        f"εr        = {state.epsr:.4f}\n"
        f"v (medio) = {v_medio:.4f}\n"
        f"n (medio) = {n_medio:.4f}\n"
        f"E(0)      = {Ez[0]:.4f}\n"
        f"H(medio)  = {Hy[len(Hy)//2]:.4f}"
    )


    # Rotación automática de la cámara (checkbox índice 4), igual idea que
    # en los otros scripts: solo gira si el usuario no la desactivó.
    if status[4]:

        ax.view_init(
            elev=ax.elev,
            azim=(ax.azim + 0.45) % 360
        )

    return (
        line_E,
        line_H,
        info
    )


# `ani` se guarda en una variable de módulo para que no sea recolectada
# por el garbage collector (si no, la animación se detendría en silencio).
ani = FuncAnimation(
    fig,
    update,
    interval=40,
    blit=False,
    cache_frame_data=False
)


if __name__ == "__main__":

    plt.show()