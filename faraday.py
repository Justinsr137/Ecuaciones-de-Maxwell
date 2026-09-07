

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
mu0 = 4 * np.pi * 1e-7


DECIMALS = 3


def fmt(x):
    return f"{x:.{DECIMALS}g}"

# GEOMETRÍA 

W, H, D = 6.0, 8.0, 2.0
W_WINDOW, H_WINDOW = 2.4, 4.4

left_thickness = (W - W_WINDOW) / 2
top_thickness = (H - H_WINDOW) / 2

x_left = -(W_WINDOW / 2 + left_thickness / 2)
x_right = +(W_WINDOW / 2 + left_thickness / 2)


def cuboid(x0, x1, y0, y1, z0, z1):
    v = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])
    return [
        [v[0], v[1], v[2], v[3]], [v[4], v[5], v[6], v[7]],
        [v[0], v[1], v[5], v[4]], [v[2], v[3], v[7], v[6]],
        [v[1], v[2], v[6], v[5]], [v[0], v[3], v[7], v[4]]
    ]


core_faces = []
core_faces += cuboid(-W / 2, -W_WINDOW / 2, -D / 2, D / 2, -H / 2, H / 2)  # Columna Izquierda
core_faces += cuboid(W_WINDOW / 2, W / 2, -D / 2, D / 2, -H / 2, H / 2)  # Columna Derecha
core_faces += cuboid(-W_WINDOW / 2, W_WINDOW / 2, -D / 2, D / 2, H_WINDOW / 2, H / 2)  
core_faces += cuboid(-W_WINDOW / 2, W_WINDOW / 2, -D / 2, D / 2, -H / 2, -H_WINDOW / 2)  


#  MODELO FÍSICO 

class IdealTransformer:
    def __init__(self):
        self.N1, self.N2 = 100, 50
        self.f, self.V0, self.RL = 1.0, 8.0, 20.0
        self.mu_r = 10000.0  # Alta permeabilidad para aproximar mu -> inf
        self.k = 0.9999  # Acoplamiento magnético casi perfecto (Ideal)
        self.R1, self.R2 = 0.1, 0.1  # Resistencias de devanado despreciables
        self.dt = 5e-4
        self.t, self.i1, self.i2 = 0.0, 0.0, 0.0
        self.phi, self.dphi_dt = 0.0, 0.0

    def core_area(self):
        return left_thickness * D

    def magnetic_path_length(self):
        return 2 * (W - left_thickness) + 2 * (H - top_thickness)

    def reluctance(self):
        return self.magnetic_path_length() / (mu0 * self.mu_r * self.core_area())

    def inductances(self):
        Rm = self.reluctance()
        L1 = self.N1 ** 2 / Rm
        L2 = self.N2 ** 2 / Rm
        M = self.k * np.sqrt(L1 * L2)
        return L1, L2, M

# Método de solución

    def step(self):

        L1, L2, M = self.inductances()
        L_mat = np.array([[L1, M], [M, L2]])
        R_diag = np.array([self.R1, self.R2 + self.RL])
        dt = self.dt

        t_next = self.t + dt
        v1 = self.V0 * np.sin(2 * np.pi * self.f * t_next)

        A = L_mat / dt + np.diag(R_diag)
        b = L_mat @ np.array([self.i1, self.i2]) / dt + np.array([v1, 0.0])

        det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
        i1_new = (b[0] * A[1, 1] - A[0, 1] * b[1]) / det
        i2_new = (A[0, 0] * b[1] - A[1, 0] * b[0]) / det

        self.i1, self.i2, self.t = i1_new, i2_new, t_next

        Rm = self.reluctance()
        phi_new = (self.N1 * self.i1 + self.N2 * self.i2) / Rm
        self.dphi_dt = (phi_new - self.phi) / dt  # diferencia finita hacia atrás para dPhi/dt
        self.phi = phi_new
        return v1, -self.RL * self.i2, phi_new

    def reset(self):
        self.t = self.i1 = self.i2 = self.phi = self.dphi_dt = 0.0


state = IdealTransformer()

# GEOMETRÍA DE LAS BOBINAS, LÍNEAS B Y ANILLOS DE CAMPO E

coil_z0, coil_z1 = -H_WINDOW / 2 * 0.86, +H_WINDOW / 2 * 0.86
coil_rx, coil_ry = left_thickness * 0.62, D * 0.68


def make_coil(x_center, turns):
    visual_turns = min(max(int(turns), 5), 180)
    theta = np.linspace(0, 2 * np.pi * visual_turns, visual_turns * 16)
    z = np.linspace(coil_z0, coil_z1, len(theta))
    x = x_center + coil_rx * np.cos(theta)
    y = coil_ry * np.sin(theta)
    return x, y, z


def generate_ideal_core_lines(num_density=4):
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

# Núcleo Translúcido
core_collection = Poly3DCollection(core_faces, facecolor=CORE_COLOR, edgecolor="#333333", alpha=0.25, linewidth=0.5)
ax.add_collection3d(core_collection)

# Bobinas
x1, y1, z1 = make_coil(x_left, state.N1)
x2, y2, z2 = make_coil(x_right, state.N2)
coil_primary, = ax.plot(x1, y1, z1, color=COIL_PRIMARY, linewidth=2.5, label="Devanado Primario")
coil_secondary, = ax.plot(x2, y2, z2, color=COIL_SECONDARY, linewidth=2.5, label="Devanado Secundario")

# Líneas del Campo Magnético Ideal (Verdes dentro del núcleo)
B_core_lines = []
for path in core_B_paths:
    line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=CORE_B_COLOR, linewidth=2.2, alpha=0.9)
    B_core_lines.append(line)

# Líneas del Campo Eléctrico Inducido E (Azul Cian envolviendo el núcleo)
E_ring_lines = []
for path in E_rings_paths:
    line, = ax.plot(path[:, 0], path[:, 1], path[:, 2], color=E_FIELD_COLOR, linewidth=2.0, alpha=0.8)
    E_ring_lines.append(line)

arrow_objects = []

# Encabezado bajado un poco (pad reducido) para que no quede pegado al borde superior
ax.set_title("LÍNEAS DE CAMPO MAGNÉTICO ($\mathbf{B}$) Y ELÉCTRICO ($\mathbf{E}$)", color=TITLE_COLOR, fontsize=16,
             pad=4, y=1.03)

# CONTROLES Y TEXTO

info_ax = fig.add_axes([0.66, 0.28, 0.30, 0.24], facecolor=PANEL_BG)
info_ax.set_xticks([]);
info_ax.set_yticks([])
for spine in info_ax.spines.values(): spine.set_color(LINE_PANEL)

info = info_ax.text(0.05, 0.92, "", transform=info_ax.transAxes, color=TEXT, fontsize=10, family="monospace", va="top")

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

# Panel "CAMPOS Y VISTA": botones reales de encendido/apagado (en vez de casillas)
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
    return f"{name}  [{'ON' if toggle_states[key] else 'OFF'}]"


def _make_toggle(key, name, on_color, y):
    btn_ax = fig.add_axes([panel_x + 0.015, y, panel_w - 0.03, 0.055])
    is_on = toggle_states[key]
    btn = Button(btn_ax, _btn_label(key, name), color=on_color if is_on else OFF_COLOR,
                hovercolor=on_color)
    btn.label.set_color(TEXT if is_on else "#777777")
    btn.label.set_fontsize(10.5)

    def on_click(event, key=key, name=name, on_color=on_color, btn=btn):
        toggle_states[key] = not toggle_states[key]
        is_on = toggle_states[key]
        btn.color = on_color if is_on else OFF_COLOR
        btn.ax.set_facecolor(btn.color)
        btn.label.set_text(_btn_label(key, name))
        btn.label.set_color(TEXT if is_on else "#777777")
        fig.canvas.draw_idle()

    btn.on_clicked(on_click)
    toggle_buttons[key] = btn


_ys = np.linspace(panel_top - 0.10, panel_bottom + 0.03, len(toggle_specs))
for (key, name, color), y in zip(toggle_specs, _ys):
    _make_toggle(key, name, color, y)

running = [True]


def clear_arrows():
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
    if running[0]:
        for _ in range(30):
            v1, v2, phi = state.step()
    else:
        v1, v2, phi = state.V0 * np.sin(2 * np.pi * state.f * state.t), -state.RL * state.i2, state.phi

    # Amplitudes de referencia para la intensidad
    max_phi_ref = state.V0 / (2 * np.pi * max(state.f, 0.1) * max(state.N1, 1) * state.reluctance())
    b_strength = np.clip(abs(phi) / max_phi_ref, 0.05, 1.0)

    # El campo E depende del ritmo de cambio (dphi/dt)
    e_strength = np.clip(abs(state.dphi_dt) / (max_phi_ref * 2 * np.pi * max(state.f, 0.1)), 0.05, 1.0)

    # Actualizar Campo Magnético B
    for line in B_core_lines:
        line.set_visible(toggle_states["B"])
        line.set_alpha(0.1 + 0.85 * b_strength)

    # Actualizar Campo Eléctrico Inducido E (Brilla con dphi/dt)
    for line in E_ring_lines:
        line.set_visible(toggle_states["E"])
        line.set_alpha(0.05 + 0.9 * e_strength)

    coil_primary.set_visible(toggle_states["Bobinas"])
    coil_secondary.set_visible(toggle_states["Bobinas"])
    core_collection.set_visible(toggle_states["Nucleo"])

    # Realce de bobinas según la magnitud de la corriente (retroalimentación visual)
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
        add_direction_arrows(core_B_paths, 1 if phi >= 0 else -1, CORE_B_COLOR, length=0.26,
                             n_arrows=14, phase=phase_B, alpha=0.3 + 0.7 * b_strength)
    if toggle_states["E"]:
        # El campo E invierte su sentido según la derivada negativa (Ley de Lenz)
        add_direction_arrows(E_rings_paths, -1 if state.dphi_dt >= 0 else 1, E_FIELD_COLOR, length=0.18,
                             n_arrows=10, phase=phase_E, alpha=0.3 + 0.7 * e_strength)

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
    state.f, state.V0, state.N1, state.N2, state.RL = s_f.val, s_V0.val, int(s_N1.val), int(s_N2.val), s_RL.val
    x1, y1, z1 = make_coil(x_left, state.N1)
    x2, y2, z2 = make_coil(x_right, state.N2)
    coil_primary.set_data_3d(x1, y1, z1)
    coil_secondary.set_data_3d(x2, y2, z2)


for s in [s_f, s_V0, s_N1, s_N2, s_RL]: s.on_changed(parameter_callback)

ani = FuncAnimation(fig, update, interval=30, blit=False, cache_frame_data=False)

if __name__ == "__main__":
    plt.show()
