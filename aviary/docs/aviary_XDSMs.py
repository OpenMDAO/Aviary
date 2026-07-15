from pyxdsm.XDSM import XDSM
import fitz  # PyMuPDF


full_xdsm = XDSM()

# 1. Diagonal blocks (Removed Input Files from the diagonal)
full_xdsm.add_system('opt', 'Optimization', 'Optimizer')
full_xdsm.add_system('pre', 'Function', 'Pre-Mission')
full_xdsm.add_system('miss', 'Function', 'Mission~Analysis')
full_xdsm.add_system('post', 'Function', 'Post-Mission')

# 2. Process Execution Flow (This draws the thin routing line!)
# Shows the loop: Optimizer -> Pre -> Miss -> Post -> back to Optimizer
full_xdsm.add_process(['opt', 'pre', 'miss', 'post', 'opt'])

# 3. full_xdsm Inputs (Routed from the left instead of taking up a matrix row)
full_xdsm.add_input('pre', 'User~Inputs')
full_xdsm.add_input('miss', 'User~Inputs')
full_xdsm.add_input('post', 'User~Inputs')

# 4. Connections from Optimizer (Using blank labels for repeated data to save space)
full_xdsm.connect('opt', 'pre', 'Design~Vars')
full_xdsm.connect('opt', 'miss', '')  # Blank, implied by the column
full_xdsm.connect('opt', 'post', '')  # Blank, implied by the column

# 5. Connections between the three phases
full_xdsm.connect('pre', 'miss', '')
full_xdsm.connect('pre', 'post', '')  # Blank, implied
full_xdsm.connect('miss', 'post', '')

# 6. Connections back to the Optimizer
full_xdsm.connect('pre', 'opt', ['Constraints,', 'Objectives'])
full_xdsm.connect('miss', 'opt', '')  # Blank
full_xdsm.connect('post', 'opt', '')  # Blank

# 7. Outputs to the right-hand side
full_xdsm.add_output('pre', 'Pre~Results', side='right')
full_xdsm.add_output('miss', 'Mission~Results', side='right')
full_xdsm.add_output('post', 'Post~Results', side='right')

full_xdsm.write('aviary_full_xdsm', build=True)

# Convert PDF to PNG
# Open the PDF file
pdf_document = fitz.open('aviary_full_xdsm.pdf')

# Load the first page (index 0)
page = pdf_document.load_page(0)

# Get a pixel map (image) of the page.
# The dpi=300 argument ensures it looks sharp in your documentation.
pix = page.get_pixmap(dpi=300)

# Save it as a PNG
pix.save('aviary_full_xdsm.png')
pdf_document.close()

simple_xdsm = XDSM()

# Use ~ to force spaces in the LaTeX output
simple_xdsm.add_system('opt', 'Optimization', 'Optimizer')
simple_xdsm.add_system('aviary', 'Function', 'AviaryProblem')

# Connections with ~ for spaces
simple_xdsm.add_input('aviary', 'User~Inputs')
simple_xdsm.connect('opt', 'aviary', 'Design~Variables')
simple_xdsm.connect('aviary', 'opt', ['Constraints,', 'Objectives'])

simple_xdsm.add_process(['opt', 'aviary', 'opt'])

# Output
simple_xdsm.add_output('aviary', 'Results', side='right')

# Generate and build
simple_xdsm.write('aviary_simple_xdsm')

# Convert PDF to PNG
# Open the PDF file
pdf_document = fitz.open('aviary_simple_xdsm.pdf')

# Load the first page (index 0)
page = pdf_document.load_page(0)

# Get a pixel map (image) of the page.
# The dpi=300 argument ensures it looks sharp in your documentation.
pix = page.get_pixmap(dpi=300)

# Save it as a PNG
pix.save('aviary_simple_xdsm.png')
pdf_document.close()
