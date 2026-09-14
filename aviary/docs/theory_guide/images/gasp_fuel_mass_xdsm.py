"""
Generate the GASP-based fuel-system XDSM diagram (gasp_fuel_mass.png).

Non-fuel inputs are abstracted as aircraft:* / mission:*. Only fuel-relevant connections
are drawn explicitly.

Requires the pyxdsm package (pip install pyxdsm), a LaTeX toolchain, and pymupdf for the
PDF -> PNG conversion.
"""

import pymupdf
from pyxdsm.XDSM import XDSM

x = XDSM()

# 1. Diagonal blocks (analysis components), in execution order.
x.add_system('fuelsys', 'Function', 'FuelSysAndFullFuselageMass')
x.add_system('fuel', 'Function', 'FuelMass')
x.add_system('fuelcomp', 'Function', 'FuelComponents')
x.add_system('wingfuelmin', 'Function', 'WingFuelMin')
x.add_system('tankcap', 'Function', 'TankCapacity')

# 2. External inputs (non-fuel inputs abstracted as aircraft:* / mission:*).
x.add_input('fuelsys', ['aircraft:*'])
x.add_input('fuel', ['aircraft:*', 'mission:*'])
x.add_input('fuelcomp', ['aircraft:*', 'mission:*'])
x.add_input('wingfuelmin', ['aircraft:*'])
x.add_input('tankcap', ['aircraft:*'])

# 3. Connections between components.
x.connect('fuelsys', 'fuel', ['aircraft:fuel:fuel\\_system:mass'])
x.connect('fuel', 'fuelcomp', ['fuel\\_mass\\_required'])
x.connect('fuel', 'tankcap', ['fuel\\_mass\\_required'])
x.connect('fuel', 'wingfuelmin', ['fuel\\_mass\\_min'])
x.connect('fuel', 'fuelsys', ['fuel\\_mass'])
x.connect(
    'fuelcomp',
    'wingfuelmin',
    ['aircraft:fuel:wing\\_volume\\_design', 'aircraft:fuel:wing\\_volume\\_structural\\_max'],
)
x.connect(
    'fuelcomp',
    'tankcap',
    [
        'aircraft:fuel:wing\\_volume\\_design',
        'max\\_wingfuel\\_mass',
        'aircraft:fuel:wing\\_volume\\_structural\\_max',
    ],
)
x.connect('wingfuelmin', 'fuelsys', 'wingfuel\\_mass\\_min')

# 5. Top-level fuel outputs (Aviary variables) to the right-hand side.
x.add_output('fuelsys', 'aircraft:fuel:fuel\\_system\\_mass', side='right')
x.add_output(
    'fuelcomp',
    ['aircraft:fuel:wing\\_volume\\_design', 'aircraft:fuel:wing\\_volume\\_structural\\_max'],
    side='right',
)
x.add_output(
    'tankcap',
    ['aircraft:fuel:auxiliary\\_fuel\\_mass\\_capacity', 'aircraft:fuel:max\\_capacity\\_mass'],
    side='right',
)

x.write('gasp_fuel_mass')

# Convert the generated PDF to PNG for the docs.
pdf_document = pymupdf.open('gasp_fuel_mass.pdf')
page = pdf_document.load_page(0)
pix = page.get_pixmap(dpi=300)
pix.save('gasp_fuel_mass.png')
pdf_document.close()
