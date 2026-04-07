import csv
# import openmdao.api as om
# from aviary.utils.aviary_values import AviaryValues


def save_to_csv_file(filename, aviary_inputs):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['name', 'value', 'units']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        for name, value_units in sorted(aviary_inputs):
            value, units = value_units
            writer.writerow({'name': name, 'value': value, 'units': units})
