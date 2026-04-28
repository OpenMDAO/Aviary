import csv
import numpy as np
import getpass
from enum import Enum
from pathlib import Path
from datetime import datetime


def save_to_csv_file(filename, aviary_inputs):
    filename = Path(filename)
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        timestamp = datetime.now().strftime('%m/%d/%y at %H:%M')
        user = getpass.getuser()

        writer.writerow([f'# created {timestamp} by {user}'])
        writer.writerow([])

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
