from pyxdsm.XDSM import FUNC, GROUP, IFUNC, XDSM
from aviary.variable_info.variables import Aircraft, Mission

x = XDSM()

HAS_STRUT = True
HAS_FOLD = True

# Create subsystem components
x.add_system("wing_mass_solve", IFUNC, ["WingMassSolve"])
x.add_system("tot_wing_mass", FUNC, ["WingMassTotal"])

# add inputs
x.add_input("wing_mass_solve", [
    Mission.Design.GROSS_MASS,
    Aircraft.Wing.HIGH_LIFT_MASS,
    Aircraft.Wing.ULTIMATE_LOAD_FACTOR,
    Aircraft.Wing.MASS_COEFFICIENT,
    Aircraft.Wing.MATERIAL_FACTOR,
    Aircraft.Engine.POSITION_FACTOR,
    Aircraft.Wing.SPAN,
    Aircraft.Wing.TAPER_RATIO,
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT,
    r"\textcolor{gray}{c_strut_braced}",
    "c_gear_loc",
    "half_sweep",
])
mass_inputs = []
if HAS_STRUT:
    mass_inputs.append(
        r"\textcolor{gray}{aircraft:strut:mass_coefficient}",
    )
if HAS_FOLD:
    mass_inputs.append(r"\textcolor{gray}{aircraft:wing:area}")
    mass_inputs.append(r"\textcolor{gray}{aircraft:wing:folding_area}")
    mass_inputs.append(r"\textcolor{gray}{aircraft:wing:fold_mass_coefficient}")

if HAS_STRUT or HAS_FOLD:
    x.add_input("tot_wing_mass", mass_inputs)

# connect up tot_wing_mass
x.connect("wing_mass_solve", "tot_wing_mass", ["isolated_wing_mass"])

mass_output = [Aircraft.Wing.MASS]
if HAS_STRUT:
    mass_output.append(
        r"\textcolor{gray}{aircraft:strut:mass}",
    )
if HAS_FOLD:
    mass_output.append(
        r"\textcolor{gray}{aircraft:wing:fold_mass}",
    )
x.add_output("tot_wing_mass", mass_output, side="right")

x.write("wing_mass_xdsm")
x.write_sys_specs("wing_mass_specs")
