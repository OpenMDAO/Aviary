import numpy as np
import os
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft
from aviary.models.external_subsystems.UAV.mass.utils.materials_database import materials
from aviary.utils.functions import get_path

"""
These are the functions currently needed to load the airfoil 
CSV into the UAV mass wing and tail components
"""


def load_airfoil_csv(file_path, delimiter=',', header=False):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Airfoil CSV file '{file_path}' not found.")

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