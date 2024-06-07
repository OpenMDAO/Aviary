import pandas as pd
import csv


# Read the CSV data into a DataFrame
df = pd.read_csv('turbofan_28k.deck')

# Calculate the electric_power column
df['electric_power (kW)'] = df[' Gross_Thrust (lbf)'] * 1.e-2

# Define the output file name
output_file = 'turbofan_28k_with_electric.deck'

# Write the DataFrame to a CSV file with the desired formatting
with open(output_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header
    writer.writerow(df.columns)

    # Write the data with formatted spacing
    for index, row in df.iterrows():
        writer.writerow(['{:>20}'.format(item) for item in row])
