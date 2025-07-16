"""
Database for various material densities that are to be used for mass calculations for small aircraft in particular.

This database will be expanded as needed.

"""

from aviary.utils.named_values import NamedValues
from aviary.utils.named_values import get_keys, get_values, get_items

materials = NamedValues()

"""
All densities below came from:
- https://tpsx.arc.nasa.gov/MaterialsDatabase 
- https://cdn.pefc.org/furniture.pefc.org/media/2023-10/784c9eb4-2f65-4465-be6e-131b0c1d5dea/84952881-7f2f-5308-8a03-99f31860c423.pdf
"""

# Wood
materials.set_val('Balsa', 130, units='kg/m**3')
materials.set_val('Cypress', 460, units='kg/m**3')
materials.set_val('Mahogany', 540, units='kg/m**3')
materials.set_val('Maple', 710, units='kg/m**3')
materials.set_val('Teak', 640, units='kg/m**3')
materials.set_val('Ply', 620, units='kg/m**3')

# Aluminum Compounds and Alloys
materials.set_val('Aluminum Oxide', 3400, units='kg/m**3')
materials.set_val('2024-T8XX', 2800, units='kg/m**3')  # aircraft-grade strength Aluminum alloy
materials.set_val('2219-T8XX', 2810, units='kg/m**3')  # Exceptionally strong Aluminum alloy
materials.set_val('2024-T6', 2770, units='kg/m**3')  # Another Aluminum alloy
materials.set_val('Aluminum Foam', 1300, units='kg/m**3')

# Steel
materials.set_val('Stainless Steel 17-4 PH', 7830, units='kg/m**3')  # 17-4 PH stainless steel
materials.set_val('Stainless Steel-AISI 302', 8060, units='kg/m**3')  # AISI 302
materials.set_val('Stainless Steel-AISI 304', 7900, units='kg/m**3')  # AISI 304
materials.set_val('Steel Alloy Cast', 7830, units='kg/m**3')  # General steel alloy cast
materials.set_val('Steel 321', 8030, units='kg/m**3')  # Steel type 321

# Carbon Fibers / Carbon - Silicon Fibers
materials.set_val('Carbon/Silicon-Carbide', 2080, units='kg/m**3')  # Carbon fiber reinforced SiC
materials.set_val(
    'Silicon-Carbide/Silicon-Carbide', 2400, units='kg/m**3'
)  # SiC fiber reinforced SiC matrix
materials.set_val('Advanced Carbon-Carbon Composite', 1600, units='kg/m**3')  # ACC
materials.set_val('Reinforced Carbon-Carbon', 1580, units='kg/m**3')
materials.set_val(
    'Reinforced Carbon-Carbon Composite', 1580, units='kg/m**3'
)  # Generally, ACC is better, but RCC is slightly cheaper

"""
Below are miscellaneous values that could be of importance, particularly for small aircraft.

These values were found from a variety of sources, and depending on the source/brand, the density 
could be slightly different. For some cases, temperature of the material also matters (typically 
the values are provided as a relative density). If there is a temperature dependence from the source,
it will be noted as a comment next to the line where the material value is set. Below are some sources 
for various values. 

The values below were not explicity listed from the above source.

Wood glue: https://www.gorillatough.com/wp-content/uploads/Gorilla-Wood-Glue-v1.2.pdf

EPS Foam: https://www.abtfoam.com/wp-content/uploads/2020/05/EPS-Standard-Sheet-Sizes-Densities-and-R-values.pdf
    Note that there is a density range given, along with different types. The density value used is for Type I, 
    and the value given is the average of the minimum and maximum within the range provided. The base unit in 
    this document is pcf for the density. It was converted to kg/m^3 for the actual value input. 

"""

materials.set_val(
    'Wood Glue', 1080, units='kg/m**3'
)  # Relative density value -- corresponds to 25 C (77 F)
materials.set_val('EPS Foam', 16.3388, units='kg/m**3')
