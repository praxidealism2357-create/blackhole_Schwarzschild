import os
import numpy as np
import matplotlib.pyplot as plt

C_VAL = 1.0
RS_VAL = 1.0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TITULOS_MODO = {
    'orbita': 'Órbita ligada',
    'caida_radial': 'Caída radial',
    'espiral': 'Caída en espiral',
}

MODOS = ['orbita', 'caida_radial', 'espiral']


def V_eff(r, L):
    """Potencial efectivo V(r) = (1 - r_s/r) (c^2 + L^2/r^2)."""
    return (1 - RS_VAL / r) * (C_VAL**2 + L**2 / r**2)


def condiciones(MODO):
    """Devuelve (r0, L, E2) con las mismas condiciones iniciales de la simulación."""
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
        return r_min, np.sqrt(L2), E2

    elif MODO == 'caida_radial':
        r0 = 10.0
        L = 0.0
    else:  # 'espiral'
        r0 = 10.0
        L = 1.8

    # Energía conservada: E^2 = V_eff(r0) cuando u^r(0) = 0
    E2 = V_eff(r0, L)
    return r0, L, E2


fig, axs = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

r = np.linspace(1.02 * RS_VAL, 25 * RS_VAL, 2000)

for ax, modo in zip(axs, MODOS):
    r0, L, E2 = condiciones(modo)
    V = V_eff(r, L)

    print(f"{modo}: r0 = {r0:.3f}, L = {L:.4f}, E^2/c^2 = {E2:.5f}")

    ax.plot(r, V, color='darkorange', lw=1.8, label=r'$V_{eff}(r)$')
    ax.axhline(E2, color='royalblue', ls='--', lw=1.3, label=r'Energía $E^2/c^2$')
    ax.axvline(RS_VAL, color='dimgray', ls='--', lw=1.2, label='Horizonte de sucesos ($r=r_s$)')
    ax.scatter([r0], [E2], color='indigo', s=60, zorder=5, label='Posición inicial')

    if modo == 'orbita':
        r_max = 20.0
        ax.scatter([r_max], [E2], color='crimson', s=50, zorder=5, label='Puntos de retorno')

    ax.set_xlabel('r')
    ax.set_title(TITULOS_MODO[modo])
    ax.set_xlim(0, 25 * RS_VAL)
    ax.set_ylim(0, 1.3)
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

axs[0].set_ylabel(r'$V_{eff}(r)$')
fig.suptitle('Potencial efectivo en la métrica de Schwarzschild')

plt.tight_layout()
output_path = os.path.join(BASE_DIR, 'potencial_efectivo_schwarzschild.png')
plt.savefig(output_path, dpi=150)
print(f"Figura guardada en {output_path}")

plt.show()