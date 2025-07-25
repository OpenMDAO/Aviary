import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def shoelace_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def compute_max_thickness(x, y):
    """
    Assumes airfoil is ordered from leading edge -> trailing edge (upper surface),
    then trailing edge -> leading edge (lower surface).
    """
    if len(x) != len(y):
        raise ValueError('x and y must be the same length.')

    # Ensure sorting in x-direction
    x_sorted = np.argsort(x)
    x = x[x_sorted]
    y = y[x_sorted]

    # Discretize and bin x
    bins = np.linspace(0, max(x), 10 * len(x))

    upper = np.interp(bins, x, y)
    lower = np.interp(bins, x, -y)
    thickness = upper - lower

    max_thickness = np.max(thickness)
    max_idx = np.argmax(thickness)
    x_location = bins[max_idx]

    return max_thickness, x_location


def visualize_airfoil_area(x, y, show=True):
    # Ensure closed loop
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    area = shoelace_area(x, y)
    max_thickness, x_thickness = compute_max_thickness(x, y)

    # Plot airfoil
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, 'k-', linewidth=2, label='Airfoil Outline')
    ax.fill(x, y, 'skyblue', alpha=0.4, label=f'Area = {area:.4f} in²')

    # Mark max thickness
    ax.axvline(
        x_thickness,
        color='purple',
        linestyle='--',
        label=f'Max Thickness = {max_thickness:.4f} in @ x={x_thickness:.2f}',
    )

    # Draw shoelace triangles (optional)
    origin = np.array([0.0, 0.0])
    for i in range(len(x) - 1):
        triangle = np.array([origin, [x[i], y[i]], [x[i + 1], y[i + 1]]])
        poly = Polygon(triangle, color='orange', alpha=0.05)
        ax.add_patch(poly)
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], 'r--', alpha=0.6)

    # Format plot
    ax.set_title('Airfoil Area and Max Thickness')
    ax.set_xlabel('x (in)')
    ax.set_ylabel('y (in)')
    ax.axis('equal')
    ax.grid(True)
    ax.legend()

    if show:
        plt.show()

    return area, max_thickness, x_thickness


def load_airfoil_dat(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y


def load_airfoil_csv(file_path, delimiter=',', header=False):
    skip = 1 if header else 0
    data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip)

    if data.shape[1] < 2:
        raise ValueError('CSV must contain at least two columns for x and y coordinates.')

    x = data[:, 0]
    y = data[:, 1]

    x_min = np.min(x)
    x_max = np.max(x)
    chord_length = x_max - x_min

    if chord_length <= 0:
        raise ValueError('Invalid airfoil: chord length must be > 0.')

    x_normalized = (x - x_min) / chord_length
    y_normalized = y / chord_length

    return x_normalized, y_normalized


# ==== Example usage ====
chord_length = 13.4  # in
x, y = load_airfoil_csv(
    r'aviary/examples/external_subsystems/dbf_based_mass/n0012-il.csv', header=True
)

area, t_max, x_t_max = visualize_airfoil_area(x, y)

area = area * (chord_length**2)

print(f'Airfoil cross-sectional area: {area:.4f} in²')

t_c_ratio = t_max / chord_length
print(f'Thickness-to-chord ratio: {t_c_ratio:.4f}')
print(f'Max thickness: {t_max:.4f} in at x = {x_t_max:.4f}')
