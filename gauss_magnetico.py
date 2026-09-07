# -*- coding: utf-8 -*-
"""
Ley de Gauss para el magnetismo -- Espira circular 3D (VERSIÓN INTERACTIVA)
=============================================================================
- Campo B: Biot-Savart discretizado sobre la espira (NumPy vectorizado).
- Líneas de campo: integradas con RUNGE-KUTTA DE 4º ORDEN (RK4) hecho a mano,
  dr/ds = B(r)/|B(r)|, en el plano meridional y luego rotadas 3D.
- Interactividad (equivalente al panel del HTML original):
    * Slider  -> Corriente I
    * Slider  -> Radio R
    * Checkboxes -> Vectores B / Líneas de campo / Flechas de corriente /
                     Esfera gaussiana / Malla de evaluación / Rotación automática
    * Botón   -> Restablecer cámara
    * Botón   -> Verificar ∇·B (recalcula max/mean/rms de la divergencia)
    * Arrastrar con el mouse sobre el gráfico 3D -> rotar la cámara libremente
      (funciona siempre que "Rotación automática" esté desactivada).

Paleta: negro y rojo dominantes. Cada elemento del campo tiene un color propio
dentro de esa familia cálida para que se distingan a simple vista:
  - Espira (fuente de corriente)........... granate oscuro
  - Flechas de corriente.................... rojo brillante
  - Líneas de campo B (RK4)................. crema/dorado claro
  - Vectores B en la malla dispersa......... naranja/dorado
  - Esfera gaussiana........................ granate translúcido

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
# 1) ESTADO FÍSICO 
# ============================================================

state = {"I": 10.0, "R": 1.5}
MU0 = 4 * np.pi * 1e-7
SEGMENTS = 220
SOFTENING = 0.055

# ============================================================
# 2) BIOT-SAVART
# ============================================================

def build_source(R):
    phi = np.linspace(0, 2 * np.pi, SEGMENTS, endpoint=False)
    dphi = 2 * np.pi / SEGMENTS
    pts = np.stack([R * np.cos(phi), R * np.sin(phi), np.zeros_like(phi)], axis=1)
    tangent = np.stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)], axis=1)
    dl = tangent * (R * dphi)
    return pts, dl

SRC_PTS, SRC_DL = build_source(state["R"])

def B_field(r):
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

def unit_tangent(r):
    b = B_field(r)
    n = np.linalg.norm(b)
    return b / n if n > 1e-14 else np.zeros(3)

def rk4_step(r, h):
    k1 = unit_tangent(r)
    k2 = unit_tangent(r + 0.5 * h * k1)
    k3 = unit_tangent(r + 0.5 * h * k2)
    k4 = unit_tangent(r + h * k3)
    return r + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

def trace_meridional_line(x0, z0, h, max_steps, R):
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
    c, s = np.cos(angle), np.sin(angle)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return points @ Rz.T

SEEDS_REF = [(1.0, 0.65), (1.25, 0.85), (1.55, 1.05),
             (1.85, 1.25), (2.2, 1.45), (2.65, 1.7)]      # calibrados para R=1.5
AZIMUTHS = np.linspace(0, 2 * np.pi, 8, endpoint=False)

def compute_field_lines(R):
    scale = R / 1.5
    h = 0.03 * R
    lines = []
    for x, z in SEEDS_REF:
        ml = trace_meridional_line(x * scale, z * scale, h, 260, R)
        for az in AZIMUTHS:
            lines.append(rotate_z(ml, az))
    return lines

# ============================================================
# 4) VERIFICACIÓN NUMÉRICA DE ∇·B
# ============================================================

def divergence_stats(R, n=9, lim=2.8, exclude=0.28):
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
loop_line, = ax.plot([], [], [], color=LOOP_COLOR, lw=3, zorder=5)

# --- Colección única para TODAS las líneas de campo (rápida de actualizar) ---
field_line_collection = Line3DCollection([], colors=FIELD_LINE_COLOR, linewidths=1.1, alpha=0.78)
ax.add_collection3d(field_line_collection, autolim=False)

# --- Artistas que SÍ se recrean en cada rebuild (quiver 3D no es "editable") ---
current_arrows_artist = [None]
vector_field_artist = [None]

# --- Elementos estáticos (no dependen de I ni R) ---
u, v = np.meshgrid(np.linspace(0, 2*np.pi, 30), np.linspace(0, np.pi, 20))
Rs = 2.6
gaussian_surface = ax.plot_surface(
    Rs*np.cos(u)*np.sin(v), Rs*np.sin(u)*np.sin(v), Rs*np.cos(v),
    color=SPHERE_COLOR, alpha=0.09, linewidth=0
)
gaussian_surface.set_visible(False)

grid_n, grid_lim = 10, 2.8
gx, gy = np.meshgrid(np.linspace(-grid_lim, grid_lim, grid_n),
                      np.linspace(-grid_lim, grid_lim, grid_n))
eval_points = ax.scatter(gx.ravel(), gy.ravel(), np.zeros(grid_n*grid_n),
                          s=6, color=EVAL_COLOR)
eval_points.set_visible(False)

# ============================================================
# 6) REBUILD
# ============================================================

def rebuild(_=None):
    I, R = state["I"], state["R"]
    global SRC_PTS, SRC_DL
    SRC_PTS, SRC_DL = build_source(R)

    # Espira
    phi = np.linspace(0, 2*np.pi, 240)
    loop_line.set_data(R*np.cos(phi), R*np.sin(phi))
    loop_line.set_3d_properties(np.zeros_like(phi))

    # Líneas de campo (RK4)
    lines = compute_field_lines(R)
    field_line_collection.set_segments(lines)

    # Flechas de corriente (recreadas: los quiver 3D no tienen "set_data")
    if current_arrows_artist[0] is not None:
        current_arrows_artist[0].remove()
    arrow_phi = np.linspace(0, 2*np.pi, 8, endpoint=False)
    px = R*np.cos(arrow_phi); py = R*np.sin(arrow_phi); pz = np.zeros_like(arrow_phi)
    tx = -np.sin(arrow_phi)*0.35; ty = np.cos(arrow_phi)*0.35; tz = np.zeros_like(arrow_phi)
    current_arrows_artist[0] = ax.quiver(px, py, pz, tx, ty, tz,
                                          color=CURRENT_COLOR, linewidth=2, arrow_length_ratio=0.55)
    current_arrows_artist[0].set_visible(check.get_status()[2])

    # Vectores B en malla dispersa (recreados)
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
    state["I"] = val
    rebuild()

def on_R(val):
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
    dmax, dmean, drms = divergence_stats(state["R"])
    stats_line[0] = f"max|∇·B| = {dmax:.2e}    mean|∇·B| = {dmean:.2e}    rms|∇·B| = {drms:.2e}"
    refresh_title()
    fig.canvas.draw_idle()

btn_stats.on_clicked(on_stats)

# ============================================================
# 8) ROTACIÓN AUTOMÁTICA FuncAnimation
#     — si está desactivada, el mouse controla la cámara libremente.
# ============================================================

def update_frame(_frame):
    if check.get_status()[5]:                 # "Rotación automática"
        az = (ax.azim + 0.6) % 360
        ax.view_init(elev=ax.elev, azim=az)
    return []

ani = animation.FuncAnimation(fig, update_frame, frames=itertools.count(),
                               interval=40, blit=False, cache_frame_data=False)

# ============================================================
# 9) INICIALIZACIÓN
# ============================================================

rebuild()
on_stats(None)   # primer cálculo de divergencia al abrir

if __name__ == "__main__":
    plt.show()
