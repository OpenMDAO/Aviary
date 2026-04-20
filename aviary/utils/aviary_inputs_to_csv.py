import csv
from enum import Enum
import numpy as np


def save_to_csv_file(filename, aviary_inputs):
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        for name, (value, units) in sorted(aviary_inputs):
            output = [name]
            if isinstance(value, (np.ndarray, list, tuple)):
                output.extend(value)
            elif isinstance(value, Enum):
                output.append(value.value)
            else:
                output.append(value)
            output.append(units)
            writer.writerow(output)
