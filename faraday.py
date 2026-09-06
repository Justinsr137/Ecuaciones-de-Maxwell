"""
Ley de Faraday — Transformador ideal (visualización de B y E inducido)
==========================================================================
Este script corresponde a la sección 5 ("Faraday induction law",
∇×E = -∂B/∂t) de la actividad "Computational electrodynamics training
guide", en su configuración (c) "ideal transformer geometry".

Enfoque del script (a diferencia de los otros tres): aquí NO se resuelve
el campo E o B en cada punto del espacio a partir de una ecuación de
campo (ni con fórmula cerrada como en Gauss, ni con FDTD como en la onda
de Ampère-Maxwell). En su lugar se usa un MODELO DE CIRCUITOS MAGNÉTICOS
CONCENTRADOS (lumped element model), el modelo estándar de ingeniería
para transformadores: el núcleo se reduce a una "reluctancia" Rm (el
equivalente magnético de una resistencia eléctrica), las bobinas a
inductancias L1, L2 acopladas por una inductancia mutua M, y se resuelve
un sistema de ecuaciones diferenciales ordinarias (no parciales) para las
corrientes i1(t), i2(t) del primario y del secundario. A partir de esas
corrientes se reconstruye el flujo magnético Φ(t) y su derivada dΦ/dt, que
es precisamente lo que la ley de Faraday relaciona con el campo eléctrico
inducido: ε = -N·dΦ/dt.

Las líneas de campo que se ven en la figura 3D (verdes dentro del núcleo
para B, cian rodeando cada columna para E) NO se calculan resolviendo
Biot-Savart o una ecuación de campo: son curvas geométricas fijas
(un lazo rectangular dentro del núcleo, anillos alrededor de cada
columna) cuya INTENSIDAD (opacidad) y cuya DIRECCIÓN (con flechas que
recorren la curva) sí se modulan en tiempo real con los resultados del
modelo de circuito. Es una elección deliberada: para un transformador
ideal (μr → ∞, acoplamiento k ≈ 1) la aproximación estándar es que B
queda completamente confinado dentro del núcleo y es aproximadamente
uniforme en cualquier sección transversal, así que dibujar su forma real
con Biot-Savart no aportaría información adicional frente al modelo de
circuito, mientras que sí conviene mostrar claramente:
  (a) que B se cierra dentro del núcleo (topología del ∇·B=0 ya visto en
      el script de la espira, aplicado aquí a un núcleo ferromagnético), y
  (b) que E se induce en anillos alrededor del núcleo, en el sentido que
      dicta la ley de Lenz según el signo de dΦ/dt.

Ecuación de circuito resuelta en cada paso de tiempo (ver `IdealTransformer.step`):

    L1 di1/dt + M di2/dt + R1 i1 = v1(t)
    M di1/dt + L2 di2/dt + (R2+RL) i2 = 0

con v1(t) = V0·sin(2π f t) la fuente en el devanado primario, RL la
resistencia de carga en el secundario, y L1, L2, M obtenidos del circuito
magnético equivalente (ley de Hopkinson): Rm = l_núcleo/(μ0·μr·A_núcleo),
L = N²/Rm, M = k·√(L1·L2).

Controles interactivos:
- Slider **f** — frecuencia de la fuente de voltaje en el primario.
- Slider **V₀** — amplitud de la fuente de voltaje.
- Slider **N₁, N₂** — número de vueltas del devanado primario/secundario
  (determina la relación de transformación N2/N1 y, junto con Rm, las
  inductancias L1, L2).
- Slider **R_Carga** — resistencia de carga conectada al secundario.
- Botones ON/OFF para mostrar/ocultar: campo B, campo E, bobinas, núcleo,
  y activar/desactivar la rotación automática de la cámara.

Requisitos:  pip install numpy matplotlib
Ejecutar:    python faraday.py
(usa el backend TkAgg, fijado explícitamente al inicio del script; si no
está disponible, instala `python3-tk` o cambia a otro backend interactivo)
"""


import numpy as np
import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# COLORES

BG = "#050303"
PANEL_BG = "#0A0404"
LINE_PANEL = "#5C1414"

TEXT = "#F2E9E7"
CORE_COLOR = "#666666"

COIL_PRIMARY = "#FF3B30"
COIL_SECONDARY = "#FF9416"

# Colores de Campos
CORE_B_COLOR = "#39FF14"  # Verde para Campo B
E_FIELD_COLOR = "#00E5FF"  # Azul Cian para Campo E Inducido

TITLE_COLOR = "#FF453A"
mu0 = 4 * np.pi * 1e-7   # permeabilidad magnética del vacío [T·m/A] (SI,
                          # a diferencia del script de la onda EM que usa
                          # unidades naturales con mu0=1)


DECIMALS = 3   # nº de cifras significativas usadas al formatear los
               # números del panel de información


def fmt(x):
    """Formatea un número con `DECIMALS` cifras significativas (no
    decimales fijos), usando el especificador 'g' de Python: evita que
    números muy pequeños o muy grandes en el panel se vean como '0.000'
    o con demasiados dígitos."""
    return f"{x:.{DECIMALS}g}"

# GEOMETRÍA DEL NÚCLEO
# Se modela un núcleo tipo "ventana" (window-type core), la forma clásica
# de un transformador: un marco rectangular hueco, con una columna a la
# izquierda (donde va el devanado primario) y otra a la derecha (devanado
# secundario), unidas arriba y abajo por dos "yugos" horizontales. W, H,
# D son el ancho, alto y profundidad del marco completo; W_WINDOW,
# H_WINDOW son el ancho y alto del hueco central (la "ventana").

W, H, D = 6.0, 8.0, 2.0
W_WINDOW, H_WINDOW = 2.4, 4.4

left_thickness = (W - W_WINDOW) / 2   # grosor de cada columna lateral
top_thickness = (H - H_WINDOW) / 2    # grosor de cada yugo horizontal

x_left = -(W_WINDOW / 2 + left_thickness / 2)   # centro de la columna izquierda
x_right = +(W_WINDOW / 2 + left_thickness / 2)  # centro de la columna derecha


def cuboid(x0, x1, y0, y1, z0, z1):
    """Devuelve las 6 caras de una caja rectangular (formato esperado por
    Poly3DCollection), igual idea que `box_faces` en el script de la onda
    EM. Se usa para construir cada uno de los 4 bloques rectangulares
    (2 columnas + 2 yugos) que forman el núcleo hueco."""
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])
    return [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]
    ]


# El núcleo completo se arma pegando 4 cajas rectangulares: dos columnas
# verticales (donde se enrollan los devanados) y dos yugos horizontales
# que las conectan arriba y abajo, formando el marco cerrado hueco típico
# de un transformador de núcleo tipo ventana.
core_faces = []
core_faces += cuboid(-W / 2, -W_WINDOW / 2, -D / 2, D / 2, -H / 2, H / 2)  # Columna Izquierda
core_faces += cuboid(W_WINDOW / 2, W / 2, -D / 2, D / 2, -H / 2, H / 2)  # Columna Derecha
core_faces += cuboid(-W_WINDOW / 2, W_WINDOW / 2, -D / 2, D / 2, H_WINDOW / 2, H / 2)  # Yugo Superior
core_faces += cuboid(-W_WINDOW / 2, W_WINDOW / 2, -D / 2, D / 2, -H / 2, -H_WINDOW / 2)  # Yugo Inferior


#  MODELO FÍSICO IDEAL

class IdealTransformer:
    """
    Modelo de circuito magnético concentrado de un transformador con
    núcleo de alta permeabilidad. En vez de resolver campos punto a
    punto, se resuelve un sistema de 2 ecuaciones diferenciales
    ordinarias (una por cada devanado) que describen cómo evolucionan
    las corrientes i1(t) (primario) e i2(t) (secundario) en el tiempo.
    """

    def __init__(self):
        self.N1, self.N2 = 100, 50      # número de vueltas primario/secundario
        self.f, self.V0, self.RL = 1.0, 8.0, 20.0   # frecuencia, amplitud de
                                                      # la fuente, resistencia de carga
        self.mu_r = 10000.0  # Alta permeabilidad para aproximar mu -> inf
                              # (un núcleo "ideal" concentra prácticamente
                              # todo el flujo dentro de sí mismo, con fuga
                              # despreciable hacia el aire)
        self.k = 0.9999  # Acoplamiento magnético casi perfecto (Ideal):
                          # k=1 significa que el 100% del flujo generado
                          # por un devanado atraviesa también el otro.
        self.R1, self.R2 = 0.1, 0.1  # Resistencias de devanado despreciables
                                      # (un transformador "ideal" no disipa
                                      # energía en el cobre, solo en la carga RL)
        self.dt = 5e-4    # paso de tiempo del integrador del circuito
        self.t, self.i1, self.i2 = 0.0, 0.0, 0.0   # tiempo y corrientes actuales
        self.phi, self.dphi_dt = 0.0, 0.0          # flujo magnético y su derivada

    def core_area(self):
        """Área de la sección transversal del núcleo (una columna), usada
        en el cálculo de la reluctancia magnética (ley de Hopkinson)."""
        return left_thickness * D

    def magnetic_path_length(self):
        """Longitud aproximada de la trayectoria magnética media que
        recorre el flujo al dar la vuelta completa por el marco del
        núcleo (las dos columnas más los dos yugos), necesaria también
        para la reluctancia."""
        return 2 * (W - left_thickness) + 2 * (H - top_thickness)

    def reluctance(self):
        """
        Reluctancia magnética del circuito: Rm = l / (μ0·μr·A), el
        análogo magnético de la resistencia eléctrica R = l/(σ·A) en la
        ley de Ohm. Cuanto mayor la permeabilidad μr del núcleo, menor la
        reluctancia, y más "fácil" resulta establecer flujo magnético con
        una corriente dada (mmf = N·i = Φ·Rm, la ley de Hopkinson, es el
        equivalente magnético de V = I·R).
        """
        return self.magnetic_path_length() / (mu0 * self.mu_r * self.core_area())

    def inductances(self):
        """
        A partir de la reluctancia Rm se obtienen las autoinductancias de
        cada devanado (L = N²/Rm, que sale de combinar Φ = N·i/Rm con el
        enlace de flujo λ = N·Φ = L·i) y la inductancia mutua M, que
        cuantifica cuánto del flujo generado por un devanado enlaza al
        otro devanado (M = k·√(L1·L2), con k el coeficiente de
        acoplamiento definido en `__init__`).
        """
        Rm = self.reluctance()
        L1 = self.N1 ** 2 / Rm
        L2 = self.N2 ** 2 / Rm
        M = self.k * np.sqrt(L1 * L2)
        return L1, L2, M

# Método de solución

    def step(self):
        """
        Avanza un paso de tiempo dt resolviendo el sistema de ecuaciones
        de circuito acopladas (forma matricial L·di/dt + R·i = v):

            [L1  M ] [di1/dt]   [R1      0    ] [i1]   [v1(t)]
            [M   L2] [di2/dt] + [0    R2+RL   ] [i2] = [ 0  ]

        Se discretiza con EULER IMPLÍCITO (también llamado "backward
        Euler"): se aproxima di/dt ≈ (i_nuevo - i_viejo)/dt y se evalúa
        v1 en el tiempo NUEVO t+dt, lo que lleva a un sistema lineal
        para las corrientes nuevas:

            (L/dt + R) · i_nuevo = L·i_viejo/dt + v(t+dt)

        Euler implícito se prefiere aquí sobre Euler explícito porque es
        incondicionalmente estable para este tipo de sistema lineal
        (importante ya que dt=5e-4 es fijo y no se ajusta a la frecuencia
        elegida por el usuario, que puede llegar a 3 Hz), a cambio de
        resolver un pequeño sistema 2x2 en cada paso en vez de una simple
        actualización explícita.

        El sistema 2x2 se resuelve "a mano" con la regla de Cramer (en
        vez de np.linalg.solve, que sería más genérico pero más lento
        para un sistema tan pequeño llamado en cada paso de la animación):
        se arma la matriz A y el vector b, se calcula el determinante
        `det`, y cada corriente nueva es un cociente de determinantes.
        """
        L1, L2, M = self.inductances()
        L_mat = np.array([[L1, M], [M, L2]])
        R_diag = np.array([self.R1, self.R2 + self.RL])
        dt = self.dt

        t_next = self.t + dt
        v1 = self.V0 * np.sin(2 * np.pi * self.f * t_next)   # fuente en el primario

        A = L_mat / dt + np.diag(R_diag)
        b = L_mat @ np.array([self.i1, self.i2]) / dt + np.array([v1, 0.0])

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        i1_new = (b[0] * A[1, 1] - A[0, 1] * b[1]) / det
        i2_new = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

        self.i1, self.i2, self.t = i1_new, i2_new, t_next

        # --- Flujo magnético y ley de Faraday --------------------------
        # La fuerza magnetomotriz total (mmf = N1·i1 + N2·i2, sumando la
        # contribución de ambos devanados) dividida entre la reluctancia
        # da el flujo instantáneo (ley de Hopkinson): Φ = mmf / Rm.
        Rm = self.reluctance()
        phi_new = (self.N1 * self.i1 + self.N2 * self.i2) / Rm
        # dΦ/dt se obtiene por diferencia finita hacia atrás (backward
        # difference) entre el flujo nuevo y el flujo del paso anterior.
        # Esta derivada es exactamente la cantidad que la LEY DE FARADAY
        # relaciona con la fem inducida en cada devanado: ε = -N·dΦ/dt
        # (ver el cálculo de emf1, emf2 en la función `update` más abajo,
        # y el uso de `dphi_dt` para modular la intensidad visual del
        # campo E inducido alrededor del núcleo).
        self.dphi_dt = (phi_new - self.phi) / dt  # diferencia finita hacia atrás para dPhi/dt
        self.phi = phi_new
        return v1, -self.RL * self.i2, phi_new

    def reset(self):
        """Reinicia el estado del transformador (corrientes, flujo y
        tiempo a cero), llamado desde el botón correspondiente de la
        interfaz (si se agrega) o al recargar el estado inicial."""
        self.t = self.i1 = self.i2 = self.phi = self.dphi_dt = 0.0


state = IdealTransformer()

# GEOMETRÍA DE LAS BOBINAS, LÍNEAS B Y ANILLOS DE CAMPO E
# Todo lo que sigue en esta sección es geometría puramente VISUAL: no
# participa en el cálculo físico (que ya quedó resuelto por
# `IdealTransformer`), solo construye las curvas 3D que se van a dibujar
# y cuya apariencia (opacidad, dirección de las flechas) se modula luego
# con los resultados de ese cálculo.

coil_z0, coil_z1 = -H_WINDOW / 2 * 0.86, +H_WINDOW / 2 * 0.86
coil_rx, coil_ry = left_thickness * 0.62, D * 0.68


def make_coil(x_center, turns):
    """
    Genera una hélice 3D alrededor del eje vertical centrada en
    `x_center` (columna izquierda o derecha del núcleo), con un número de
    "vueltas visuales" acotado entre 5 y 180 (`visual_turns`) para que la
    representación sea legible incluso si el slider de vueltas reales
    (N1 o N2) pide, por ejemplo, 150: dibujar 150 espiras completas se
    vería como una masa sólida indistinguible, así que se limita el
    número de vueltas DIBUJADAS sin que eso afecte el número de vueltas
    usado en el cálculo físico (que sigue siendo `turns` real, usado en
    `IdealTransformer`).
    """
    visual_turns = min(max(int(turns), 5), 180)
    theta = np.linspace(0, 2 * np.pi * visual_turns, visual_turns * 16)
    z = np.linspace(coil_z0, coil_z1, len(theta))
    x = x_center + coil_rx * np.cos(theta)
    y = coil_ry * np.sin(theta)
    return x, y, z


def generate_ideal_core_lines(num_density=4):
    """
    Genera un conjunto de "líneas de campo B" idealizadas: cada línea es
    un lazo RECTANGULAR CERRADO que recorre el contorno del núcleo (sube
    por la columna izquierda, cruza por el yugo superior, baja por la
    columna derecha, vuelve por el yugo inferior). Se generan varias
    copias ligeramente desplazadas en x e y (`x_offsets`, `y_offsets`)
    para dar la sensación de un HAZ de líneas llenando la sección
    transversal del núcleo, en vez de una sola línea sobre el eje central.

    Esta forma rectangular es la representación IDEALIZADA de B para un
    núcleo de alta permeabilidad: en la aproximación de "circuito
    magnético" (la misma que usa `IdealTransformer`), se asume que todo
    el flujo queda confinado dentro del núcleo y sigue su trayectoria
    media, sin líneas curvas escapando al aire — de ahí que las líneas
    tengan esquinas rectas en vez de curvarse suavemente como lo haría un
    campo calculado con Biot-Savart.
    """
    lines = []
    y_offsets = np.linspace(-D * 0.35, D * 0.35, num_density)
    x_offsets = np.linspace(-left_thickness * 0.35, left_thickness * 0.35, num_density)

    for dx in x_offsets:
        for dy in y_offsets:
            x_l = x_left + dx
            x_r = x_right - dx
            z_top = (H_WINDOW / 2 + H / 2) / 2 + dx * 0.2
            z_bot = -(H_WINDOW / 2 + H / 2) / 2 - dx * 0.2

            pts = []
            N_pts = 35
            for z in np.linspace(z_bot, z_top, N_pts): pts.append([x_l, dy, z])
            for x in np.linspace(x_l, x_r, N_pts): pts.append([x, dy, z_top])
            for z in np.linspace(z_top, z_bot, N_pts): pts.append([x_r, dy, z])
            for x in np.linspace(x_r, x_l, N_pts): pts.append([x, dy, z_bot])

            lines.append(np.array(pts))
    return lines


def generate_E_field_rings(num_rings=3, r_extra=0.4):
    """
    Genera anillos circulares alrededor de cada columna del núcleo
    (izquierda y derecha), a distintas alturas z. Estos anillos
    representan el CAMPO ELÉCTRICO INDUCIDO: la ley de Faraday en forma
    integral (∮E·dl = -dΦ/dt) dice que E circula en lazos CERRADOS
    alrededor de cualquier trayectoria que encierre un flujo magnético
    variable, y la forma más natural de esa trayectoria alrededor de una
    columna del núcleo es, precisamente, un anillo que la rodea (de ahí
    el radio `rx, ry` un poco mayor que la columna, `r_extra`, para que
    el anillo quede visualmente por fuera del núcleo, no atravesándolo).
    """
    rings = []
    theta = np.linspace(0, 2 * np.pi, 50)
    rx = left_thickness / 2 + r_extra
    ry = D / 2 + r_extra

    for x_c in [x_left, x_right]:
        for z_c in np.linspace(-H_WINDOW / 3, H_WINDOW / 3, num_rings):
            x = x_c + rx * np.cos(theta)
            y = ry * np.sin(theta)
            z = np.full_like(theta, z_c)
            pts = np.column_stack((x, y, z))
            rings.append(pts)
    return rings


core_B_paths = generate_ideal_core_lines(num_density=4)
E_rings_paths = generate_E_field_rings(num_rings=3)

#  CONFIGURACIÓN DE LA ESCENA 3D

fig = plt.figure(figsize=(20, 12), facecolor=BG)


ax = fig.add_axes([0.03, 0.28, 0.60, 0.66], projection="3d", facecolor=BG)
ax.set_axis_off()

ax.set_xlim(-W / 2, W / 2)
ax.set_ylim(-D * 1.5, D * 1.5)
ax.set_zlim(-H / 2, H / 2)
ax.set_box_aspect([W, D * 2.5, H])
ax.view_init(elev=20, azim=-50)

# Núcleo Translúcido (Poly3DCollection): se deja translúcido a propósito
# para que las líneas de campo B, que van POR DENTRO del núcleo, se
# alcancen a ver a través de él.
core_collection = Poly3DCollection(core_faces, facecolor=CORE_COLOR, edgecolor="#333333", alpha=0.25, linewidth=0.5)
ax.add_collection3d(core_collection)

# Bobinas: artistas persistentes (se regeneran con `set_data_3d` en
# `parameter_callback` cuando cambia el número de vueltas N1/N2).
x1, y1, z1 = make_coil(x_left, state.N1)
x2, y2, z2 = make_coil(x_right, state.N2)
coil_primary, = ax.plot(x1, y1, z1, color=COIL_PRIMARY, linewidth=2.5, label="Devanado Primario")
coil_secondary, = ax.plot(x2, y2, z2, color=COIL_SECONDARY, linewidth=2.5, label="Devanado Secundario")

# Líneas del Campo Magnético Ideal (Verdes dentro del núcleo): una línea
# `ax.plot` persistente por cada lazo generado en `generate_ideal_core_lines`;
# su opacidad (alpha) se modula cada frame según la intensidad del flujo.
B_core_lines = []
for path in core_B_paths:
    line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=CORE_B_COLOR, linewidth=2.2, alpha=0.9)
    B_core_lines.append(line)

# Líneas del Campo Eléctrico Inducido E (Azul Cian envolviendo el núcleo):
# misma idea, un `ax.plot` persistente por cada anillo de
# `generate_E_field_rings`, con opacidad modulada por |dΦ/dt|.
E_ring_lines = []
for path in E_rings_paths:
    line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=E_FIELD_COLOR, linewidth=2.0, alpha=0.8)
    E_ring_lines.append(line)

arrow_objects = []   # flechas de dirección sobre las líneas B y E (se
                      # recrean cada frame, ver `add_direction_arrows`)

# Encabezado bajado un poco (pad reducido) para que no quede pegado al borde superior
ax.set_title("LÍNEAS DE CAMPO MAGNÉTICO ($\mathbf{B}$) Y ELÉCTRICO ($\mathbf{E}$)", color=TITLE_COLOR, fontsize=16,
             pad=4, y=1.03)

# CONTROLES Y TEXTO

info_ax = fig.add_axes([0.66, 0.28, 0.30, 0.24], facecolor=PANEL_BG)
info_ax.set_xticks([]);
info_ax.set_yticks([])
for spine in info_ax.spines.values(): spine.set_color(LINE_PANEL)

info = info_ax.text(0.05, 0.92, "", transform=info_ax.transAxes, color=TEXT, fontsize=10, family="monospace", va="top")

# Sliders de los 5 parámetros del modelo de circuito (frecuencia,
# amplitud de la fuente, vueltas de cada devanado, resistencia de carga):
# variar cualquiera de ellos cambia L1, L2, M, Rm o la fuente v1(t), y por
# lo tanto toda la dinámica de i1(t), i2(t), Φ(t).
slider_axes = {name: fig.add_axes([0.08, y, 0.45, 0.022], facecolor=LINE_PANEL)
               for name, y in zip(["f", "V0", "N1", "N2", "RL"], [0.22, 0.17, 0.12, 0.07, 0.02])}

s_f = Slider(slider_axes["f"], "f [Hz]", 0.1, 3.0, valinit=state.f, color=COIL_PRIMARY)
s_V0 = Slider(slider_axes["V0"], "V₀ [V]", 1, 15, valinit=state.V0, color=COIL_PRIMARY)
s_N1 = Slider(slider_axes["N1"], "N₁ [Vueltas]", 20, 150, valinit=state.N1, valstep=5, color=COIL_PRIMARY)
s_N2 = Slider(slider_axes["N2"], "N₂ [Vueltas]", 20, 150, valinit=state.N2, valstep=5, color=COIL_SECONDARY)
s_RL = Slider(slider_axes["RL"], "R_Carga [Ω]", 5, 60, valinit=state.RL, color=COIL_SECONDARY)

for s in [s_f, s_V0, s_N1, s_N2, s_RL]:
    s.label.set_color(TEXT);
    s.valtext.set_color(TEXT)

# Panel "CAMPOS Y VISTA": botones reales de encendido/apagado (en vez de
# casillas de verificación como en los otros scripts). Aquí se optó por
# `Button` normales en vez de `CheckButtons` para poder mostrar el estado
# ON/OFF directamente en el TEXTO del botón (p. ej. "Campo B [ON]") y
# cambiar su COLOR de fondo según el estado, algo que CheckButtons no
# permite personalizar tan fácilmente. `toggle_states` es el diccionario
# de estado (equivalente a `check.get_status()` en los otros scripts,
# pero como diccionario con nombres en vez de una lista de booleanos por
# índice), y `toggle_specs` empareja cada clave con su etiqueta visible y
# su color "encendido".
toggle_states = {"B": True, "E": True, "Bobinas": True, "Nucleo": True, "Rot3D": False}
toggle_specs = [
    ("B", "Campo B", CORE_B_COLOR),
    ("E", "Campo E", E_FIELD_COLOR),
    ("Bobinas", "Bobinas", COIL_PRIMARY),
    ("Nucleo", "Núcleo", CORE_COLOR),
    ("Rot3D", "Rotación 3D", TEXT),
]
OFF_COLOR = "#1A1A1A"

panel_x, panel_w = 0.66, 0.30
panel_top, panel_bottom = 0.93, 0.55
fig.text(panel_x + panel_w / 2, panel_top, "CAMPOS Y VISTA", color=TITLE_COLOR,
         fontsize=12, ha="center", va="top")

toggle_buttons = {}


def _btn_label(key, name):
    """Construye el texto del botón incluyendo su estado actual, p. ej.
    'Campo B  [ON]' o 'Núcleo  [OFF]'."""
    return f"{name}  [{'ON' if toggle_states[key] else 'OFF'}]"


def _make_toggle(key, name, on_color, y):
    """
    Crea un botón-interruptor (toggle) individual en la posición vertical
    `y` del panel. A diferencia de un `Button` normal de un solo uso,
    aquí el propio callback `on_click` invierte el booleano en
    `toggle_states[key]` y reconstruye la apariencia del botón (color de
    fondo, color y texto de la etiqueta) para reflejar el nuevo estado, 
    simulando así el comportamiento de una casilla de verificación con
    apariencia de botón.
    """
    btn_ax = fig.add_axes([panel_x + 0.015, y, panel_w - 0.03, 0.055])
    is_on = toggle_states[key]
    btn = Button(btn_ax, _btn_label(key, name), color=on_color if is_on else OFF_COLOR,
                hovercolor=on_color)
    btn.label.set_color(TEXT if is_on else "#777777")
    btn.label.set_fontsize(10.5)

    def on_click(event, key=key, name=name, on_color=on_color, btn=btn):
        # Los argumentos con valor por defecto (key=key, etc.) son un
        # truco estándar de Python para "capturar" el valor actual de
        # estas variables en el momento en que se define la función,
        # evitando el problema clásico de los closures en un bucle (si no
        # se hiciera así, todos los botones terminarían compartiendo la
        # última key/name/on_color del bucle `for` de más abajo).
        toggle_states[key] = not toggle_states[key]
        is_on = toggle_states[key]
        btn.color = on_color if is_on else OFF_COLOR
        btn.ax.set_facecolor(btn.color)
        btn.label.set_text(_btn_label(key, name))
        btn.label.set_color(TEXT if is_on else "#777777")
        fig.canvas.draw_idle()

    btn.on_clicked(on_click)
    toggle_buttons[key] = btn


# Distribuye verticalmente los 5 botones dentro del panel y los crea.
_ys = np.linspace(panel_top - 0.10, panel_bottom + 0.03, len(toggle_specs))
for (key, name, color), y in zip(toggle_specs, _ys):
    _make_toggle(key, name, color, y)

running = [True]   # bandera de pausa (ver `update`); este script no
                    # incluye un botón "Pausa" como el de la onda EM, pero
                    # se deja la misma estructura de código por si se
                    # quiere añadir uno más adelante.


def clear_arrows():
    """Elimina todas las flechas de dirección dibujadas en el frame
    anterior (ver `add_direction_arrows`). Se llama al inicio de cada
    frame antes de volver a dibujarlas con la fase/intensidad actuales."""
    for obj in arrow_objects:
        try:
            obj.remove()
        except:
            pass
    arrow_objects.clear()


def add_direction_arrows(paths, direction, color, length=0.3, n_arrows=3, phase=0.0, alpha=1.0):
    """
    Coloca varias flechas por cada línea de campo, distribuidas a lo largo
    del recorrido cerrado. 'phase' desplaza la posición de las flechas con
    el tiempo, dando la sensación de flujo circulando por el campo.

    Detalle de implementación: para cada línea (`path`, un arreglo de
    puntos que recorre el lazo cerrado) se eligen `n_arrows` posiciones
    equiespaciadas (parametrizadas por `frac` entre 0 y 1) y se calcula la
    dirección TANGENTE a la curva en ese punto como la diferencia entre
    dos puntos consecutivos del muestreo (`p_next - p`), normalizada. El
    parámetro `direction` (+1 o -1) invierte el sentido de todas las
    flechas de una vez, que es como se implementa físicamente el cambio
    de sentido del campo según el signo de Φ (para B) o de dΦ/dt (para
    E, por la ley de Lenz) sin tener que generar geometría nueva.
    """
    for index, path in enumerate(paths):
        L = len(path)
        for k in range(n_arrows):
            frac = ((k / n_arrows) + phase) % 1.0
            j = int(frac * L)
            j_next = (j + 2) % L
            p, p_next = path[j], path[j_next]
            tangent = p_next - p
            norm = np.linalg.norm(tangent)
            if norm == 0: continue
            tangent = (tangent / norm) * direction
            q = ax.quiver(p[0], p[1], p[2], tangent[0], tangent[1], tangent[2], length=length, normalize=True,
                          color=color, arrow_length_ratio=0.6, alpha=alpha)
            arrow_objects.append(q)


# ===============================================================
# 7. ANIMACIÓN INTEGRADA
# ===============================================================
def update(frame):
    """
    Callback de animación (~33 fps, interval=30 ms). Combina en un solo
    lugar: (a) avanzar el modelo de circuito, (b) traducir sus resultados
    en la apariencia visual de las líneas de campo, y (c) actualizar el
    panel de texto con las cantidades derivadas (incluida la fem de
    Faraday).
    """
    if running[0]:
        # Se avanzan 30 sub-pasos del circuito (dt=5e-4) por cada frame
        # dibujado, misma razón que en el script de la onda EM: el paso
        # de integración es pequeño (por estabilidad/precisión) y hay que
        # acumular varios para que la animación no se vea congelada.
        for _ in range(30):
            v1, v2, phi = state.step()
    else:
        # Si estuviera en pausa (running[0] = False; actualmente no hay
        # ningún botón en este script que lo active), se recalculan v1 y
        # v2 con el último estado conocido, sin avanzar el tiempo ni
        # volver a resolver el circuito.
        v1, v2, phi = state.V0 * np.sin(2 * np.pi * state.f * state.t), -state.RL * state.i2, state.phi

    # --- Amplitudes de referencia para la intensidad visual -----------
    # `max_phi_ref` es una ESTIMACIÓN del flujo pico esperado en régimen
    # senoidal permanente (a partir de V0, f, N1 y la reluctancia), NO un
    # valor calculado exactamente del circuito. Se usa solo para
    # normalizar |Φ| a un rango [0,1] y así modular la opacidad de las
    # líneas B de forma consistente sin importar la combinación de
    # sliders elegida (si no se normalizara así, con V0 pequeño las
    # líneas casi no se verían, y con V0 grande estarían siempre al
    # máximo de opacidad).
    max_phi_ref = state.V0 / (2 * np.pi * max(state.f, 0.1) * max(state.N1, 1) * state.reluctance())
    b_strength = np.clip(abs(phi) / max_phi_ref, 0.05, 1.0)

    # El campo E depende del ritmo de cambio (dphi/dt): igual que arriba
    # pero normalizando |dΦ/dt| en vez de |Φ|, ya que por la ley de
    # Faraday es la DERIVADA del flujo (no el flujo en sí) la que induce E.
    e_strength = np.clip(abs(state.dphi_dt) / (max_phi_ref * 2 * np.pi * max(state.f, 0.1)), 0.05, 1.0)

    # Actualizar Campo Magnético B: más flujo instantáneo => líneas más
    # opacas (más "brillantes"), simulando un campo más intenso.
    for line in B_core_lines:
        line.set_visible(toggle_states["B"])
        line.set_alpha(0.1 + 0.85 * b_strength)

    # Actualizar Campo Eléctrico Inducido E (Brilla con dphi/dt): esta es
    # la representación visual directa de la ley de Faraday -- el campo E
    # inducido "brilla" más cuanto más rápido cambia el flujo, no cuanto
    # mayor es el flujo mismo (un flujo constante, por grande que sea, NO
    # induce ningún campo eléctrico).
    for line in E_ring_lines:
        line.set_visible(toggle_states["E"])
        line.set_alpha(0.05 + 0.9 * e_strength)

    coil_primary.set_visible(toggle_states["Bobinas"])
    coil_secondary.set_visible(toggle_states["Bobinas"])
    core_collection.set_visible(toggle_states["Nucleo"])

    # Realce de bobinas según la magnitud de la corriente (retroalimentación
    # visual): las corrientes de referencia (i1_ref, i2_ref) son estimaciones
    # groseras del orden de magnitud esperado (no valores físicos exactos),
    # usadas únicamente para normalizar el grosor/opacidad de la línea de
    # cada bobina entre un mínimo y un máximo visualmente razonables.
    i1_ref = max(state.V0 / max(state.R1, 1e-9) * 0.02, 1e-6)
    i2_ref = max(state.V0 / max(state.RL, 1e-9) * 1.5, 1e-6)
    coil_primary.set_linewidth(1.5 + 3.0 * np.clip(abs(state.i1) / i1_ref, 0.0, 1.0))
    coil_secondary.set_linewidth(1.5 + 3.0 * np.clip(abs(state.i2) / i2_ref, 0.0, 1.0))
    coil_primary.set_alpha(0.5 + 0.5 * np.clip(abs(state.i1) / i1_ref, 0.0, 1.0))
    coil_secondary.set_alpha(0.5 + 0.5 * np.clip(abs(state.i2) / i2_ref, 0.0, 1.0))

    # Flechas del Campo B y Campo E siguiendo TODO el recorrido de cada línea,
    # con una ligera fase animada para dar sensación de flujo circulando.
    clear_arrows()
    phase_B = (state.f * state.t * 0.15) % 1.0
    phase_E = (state.f * state.t * 0.15 + 0.5) % 1.0  # desfasada respecto a B
    if toggle_states["B"]:
        # El sentido de las flechas de B sigue el signo del flujo Φ (si
        # Φ es negativo, el flujo circula en sentido contrario al de
        # referencia, y las flechas se invierten con `direction=-1`).
        add_direction_arrows(core_B_paths, 1 if phi >= 0 else -1, CORE_B_COLOR, length=0.26,
                             n_arrows=14, phase=phase_B, alpha=0.3 + 0.7 * b_strength)
    if toggle_states["E"]:
        # El campo E invierte su sentido según la derivada negativa (Ley de Lenz):
        # esta es la aplicación explícita de la ley de Lenz (el signo
        # menos en ε=-N·dΦ/dt): el campo/corriente inducidos se oponen
        # SIEMPRE al cambio de flujo que los origina, de ahí que la
        # dirección de las flechas de E dependa del SIGNO de dΦ/dt con un
        # signo invertido respecto a las de B.
        add_direction_arrows(E_rings_paths, -1 if state.dphi_dt >= 0 else 1, E_FIELD_COLOR, length=0.18,
                             n_arrows=10, phase=phase_E, alpha=0.3 + 0.7 * e_strength)

    # --- Fuerza electromotriz inducida (ley de Faraday) ----------------
    # ε = -N·dΦ/dt para cada devanado: esta es la cantidad que la sección
    # 5.3 de la guía pide evaluar explícitamente ("Evaluate induced
    # electromotive force"). Nótese que emf1/emf2 son cantidades
    # derivadas que se muestran en el panel informativo, no se usan para
    # recalcular las corrientes (que ya vienen resueltas directamente del
    # sistema de circuito en `IdealTransformer.step`); sirven para
    # conectar explícitamente el resultado numérico con el enunciado de
    # la ley de Faraday.
    emf1 = -state.N1 * state.dphi_dt
    emf2 = -state.N2 * state.dphi_dt

    info.set_text(
        f"TRANSFORMADOR IDEAL \n"
        f"-----------------------------------------------------\n"
        f"Tiempo    = {fmt(state.t)} s\n"
        f"Relación N2/N1 = {fmt(state.N2 / state.N1)}\n"
        f"V1       = {fmt(v1)} V\n"
        f"V2       = {fmt(v2)} V\n"
        f"Flujo Φ  = {fmt(phi * 1e3)} mWb\n"
        f"dΦ/dt    = {fmt(state.dphi_dt * 1e3)} mWb/s\n"
        f"ε1 (FEM) = {fmt(emf1)} V\n"
        f"ε2 (FEM) = {fmt(emf2)} V"
    )

    if toggle_states["Rot3D"]:
        ax.view_init(elev=ax.elev, azim=(ax.azim + 0.4) % 360)

    return B_core_lines + E_ring_lines + [info, coil_primary, coil_secondary]


def parameter_callback(val):
    """Callback común a los 5 sliders: vuelca sus valores a `state` (nótese
    que N1 y N2 se convierten explícitamente a `int`, ya que el número de
    vueltas no tiene sentido como valor fraccionario) y RECONSTRUYE la
    geometría de las bobinas (`make_coil`) porque, a diferencia de f, V0
    o RL, cambiar N1/N2 sí cambia la forma visible de la hélice (más o
    menos vueltas dibujadas)."""
    state.f, state.V0, state.N1, state.N2, state.RL = s_f.val, s_V0.val, int(s_N1.val), int(s_N2.val), s_RL.val
    x1, y1, z1 = make_coil(x_left, state.N1)
    x2, y2, z2 = make_coil(x_right, state.N2)
    coil_primary.set_data_3d(x1, y1, z1)
    coil_secondary.set_data_3d(x2, y2, z2)


for s in [s_f, s_V0, s_N1, s_N2, s_RL]: s.on_changed(parameter_callback)

# `ani` se guarda en una variable de módulo para que no sea recolectada
# por el garbage collector (si no, la animación se detendría en silencio).
ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()