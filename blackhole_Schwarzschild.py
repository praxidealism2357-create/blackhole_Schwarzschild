import os
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.animation as animation


t, r, theta, phi = sp.symbols('t r theta phi')
c, r_s = sp.symbols('c r_s', positive=True)
coords = [t, r, theta, phi]

f = 1 - r_s / r
g = sp.diag(
    -f * c**2,
    1 / f,
    r**2,
    r**2 * sp.sin(theta)**2
)

g_inv = g.inv()
n = 4

christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)
for i in range(n):
    for j in range(n):
        for k in range(n):
            s = 0
            for l in range(n):
                s += g_inv[i, l] * (
                    sp.diff(g[l, j], coords[k]) +
                    sp.diff(g[l, k], coords[j]) -
                    sp.diff(g[j, k], coords[l])
                )
            christoffel[i, j, k] = sp.simplify(s / 2)

g_func = sp.lambdify((c, r_s, r, theta), christoffel.tolist(), modules='numpy')

C_VAL = 1.0
RS_VAL = 1.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TITULOS_MODO = {
    'orbita': ' (órbita ligada)',
    'caida_radial': ' (caída radial)',
    'espiral': ' (caída en espiral)',
}

MODOS = ['orbita', 'caida_radial', 'espiral']


def geodesic_rhs(s, y):
    u = y[0:4]
    v = y[4:8]

    chris = np.array(g_func(C_VAL, RS_VAL, u[1], u[2]), dtype=float)

    du = v
    dv = -np.einsum('ijk,j,k->i', chris, v, v)

    return np.concatenate((du, dv))


def cruza_horizonte(s, y):
    return y[1] - RS_VAL * 1.001


cruza_horizonte.terminal = True
cruza_horizonte.direction = -1


animaciones = []
def simular(MODO):
    print(f"\n=== Modo: {MODO} ===")

    theta0 = np.pi / 2
    phi0 = 0.0

    if MODO == 'orbita':
        r_min = 8.0
        r_max = 20.0

        def _a(rr):
            return (1 - RS_VAL / rr) / rr**2

        def _b(rr):
            return (1 - RS_VAL / rr) * C_VAL**2

        _A = np.array([[1, -_a(r_min)], [1, -_a(r_max)]])
        _B = np.array([_b(r_min), _b(r_max)])
        E2, L2 = np.linalg.solve(_A, _B)
        L = np.sqrt(L2)

        r0 = r_min
        f0 = 1 - RS_VAL / r0

        uphi0 = L / r0**2
        ur0 = 0.0
        utheta0 = 0.0
        ut0 = np.sqrt((C_VAL**2 + r0**2 * uphi0**2) / f0)

    elif MODO == 'caida_radial':
        r0 = 10.0
        f0 = 1 - RS_VAL / r0

        ut0 = C_VAL / np.sqrt(f0)
        ur0 = 0.0
        utheta0 = 0.0
        uphi0 = 0.0

    else:  # 'espiral'
        r0 = 10.0
        f0 = 1 - RS_VAL / r0
        L = 1.8

        uphi0 = L / r0**2
        ur0 = 0.0
        utheta0 = 0.0
        ut0 = np.sqrt((C_VAL**2 + r0**2 * uphi0**2) / f0)

    initial = [0.0, r0, theta0, phi0, ut0, ur0, utheta0, uphi0]

    if MODO == 'orbita':
        T = 3000
    elif MODO == 'caida_radial':
        T = 100
    else:
        T = 150

    sol = solve_ivp(
        geodesic_rhs,
        [0, T],
        initial,
        t_eval=np.linspace(0, T, 30000),
        method='RK45',
        rtol=1e-10,
        atol=1e-10,
        events=cruza_horizonte,
    )

    r_coord = sol.y[1]
    theta_coord = sol.y[2]
    phi_coord = sol.y[3]

    x = r_coord * np.sin(theta_coord) * np.cos(phi_coord)
    y_cart = r_coord * np.sin(theta_coord) * np.sin(phi_coord)

    # ----------------------------- Figura estática -----------------------------
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))

    axs[0].plot(x, y_cart, lw=1.2, color='royalblue')
    axs[0].scatter([0], [0], color='black', s=60, label='Singularidad')
    horizonte = plt.Circle((0, 0), RS_VAL, color='dimgray', alpha=0.4, label='Horizonte de sucesos')
    axs[0].add_patch(horizonte)
    axs[0].set_xlabel('x')
    axs[0].set_ylabel('y')
    axs[0].set_title('Trayectoria de la partícula en el plano ecuatorial' + TITULOS_MODO[MODO])
    axs[0].set_aspect('equal')
    if MODO == 'caida_radial':
        axs[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=9)
    else:
        axs[0].legend()
    axs[0].grid(alpha=0.3)

    axs[1].plot(sol.t, r_coord, color='darkorange')
    axs[1].axhline(RS_VAL, color='dimgray', ls='--', label='Horizonte de sucesos ($r=r_s$)')
    axs[1].set_xlabel('parámetro afín s')
    axs[1].set_ylabel('r(s)')
    axs[1].set_title('Evolución radial de la partícula')
    axs[1].legend()
    axs[1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(BASE_DIR, f'geodesica_schwarzschild_{MODO}.png')
    plt.savefig(output_path, dpi=150)

    # -------------------------------- Animación --------------------------------
    print("Generando animación.")

    N_FRAMES = 120
    n_puntos = sol.t.size
    step = max(1, n_puntos // N_FRAMES)
    idx = np.arange(0, n_puntos, step)

    fig_anim, axs_anim = plt.subplots(1, 2, figsize=(10, 4.5), dpi=80)

    axs_anim[0].plot(x, y_cart, lw=1.0, color='royalblue', alpha=0.5)
    axs_anim[0].scatter([0], [0], color='black', s=60, label='Singularidad')
    horizonte_anim = plt.Circle((0, 0), RS_VAL, color='dimgray', alpha=0.4, label='Horizonte de sucesos')
    axs_anim[0].add_patch(horizonte_anim)
    axs_anim[0].set_xlabel('x')
    axs_anim[0].set_ylabel('y')
    axs_anim[0].set_title('Trayectoria de la partícula en el plano ecuatorial' + TITULOS_MODO[MODO], fontsize=11)
    axs_anim[0].set_aspect('equal')
    if MODO == 'caida_radial':
        axs_anim[0].legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0, fontsize=8)
    else:
        axs_anim[0].legend(loc='upper right')
    axs_anim[0].grid(alpha=0.3)
    punto_orbita, = axs_anim[0].plot([], [], 'o', color='indigo', markersize=8, zorder=5)

    axs_anim[1].plot(sol.t, r_coord, color='darkorange', alpha=0.5)
    axs_anim[1].axhline(RS_VAL, color='dimgray', ls='--', label='Horizonte de sucesos ($r=r_s$)')
    axs_anim[1].set_xlabel('parámetro afín s')
    axs_anim[1].set_ylabel('r(s)')
    axs_anim[1].set_title('Evolución radial de la partícula')
    axs_anim[1].legend(loc='upper right')
    axs_anim[1].grid(alpha=0.3)
    punto_r, = axs_anim[1].plot([], [], 'o', color='indigo', markersize=8, zorder=5)

    plt.tight_layout()

    def _init():
        punto_orbita.set_data([], [])
        punto_r.set_data([], [])
        return punto_orbita, punto_r

    def _update(frame):
        i = idx[frame]
        punto_orbita.set_data([x[i]], [y_cart[i]])
        punto_r.set_data([sol.t[i]], [r_coord[i]])
        return punto_orbita, punto_r

    anim = animation.FuncAnimation(
        fig_anim, _update, frames=len(idx),
        init_func=_init, blit=True, interval=40,
    )

    gif_path = os.path.join(BASE_DIR, f'geodesica_schwarzschild_{MODO}.gif')
    anim.save(gif_path, writer=animation.PillowWriter(fps=25))
    print(f"GIF guardado en {gif_path}")

    animaciones.append(anim)


for modo in MODOS:
    simular(modo)

plt.show()
