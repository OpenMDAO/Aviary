import aviary.api as av
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_variables import Aircraft

ExtendedMetaData = av.CoreMetaData

##### WING VALUES #####
# Metadata registration for DBF custom structural parameters
av.add_meta_data(
    Aircraft.Wing.NUM_SPARS,
    units='unitless',
    desc='Number of wing spars',
    default_value=2,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.SPAR_THICKNESS,
    units='in',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.25,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.SPAR_DENSITY,
    units='lbm/in**3',
    desc='Material density of the spar',
    default_value=0.015,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.RIB_THICKNESS,
    units='in',
    desc='Thickness of a single rib',
    default_value=0.125,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.RIB_DENSITY,
    units='lbm/in**3',
    desc='Material density of the rib',
    default_value=0.012,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.SKIN_DENSITY,
    units='lbm/in**2',
    desc='Surface density of wing skin',
    default_value=0.02,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.AIRFOIL_X_COORDS,
    units='unitless',
    desc='List of the x-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Wing.AIRFOIL_Y_COORDS,
    units='unitless',
    desc='List of the y-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

##### HORIZONTAL TAIL VALUES #####
# Metadata registration for DBF custom structural parameters
av.add_meta_data(
    Aircraft.HorizontalTail.NUM_SPARS,
    units='unitless',
    desc='Number of tail spars',
    default_value=2,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SPAR_THICKNESS,
    units='in',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.25,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SPAR_DENSITY,
    units='lbm/in**3',
    desc='Material density of the spar',
    default_value=0.015,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.RIB_THICKNESS,
    units='in',
    desc='Thickness of a single rib',
    default_value=0.125,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.RIB_DENSITY,
    units='lbm/in**3',
    desc='Material density of the rib',
    default_value=0.012,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.SKIN_DENSITY,
    units='lbm/in**2',
    desc='Surface density of tail skin',
    default_value=0.02,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.AIRFOIL_X_COORDS,
    units='unitless',
    desc='List of the x-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.HorizontalTail.AIRFOIL_Y_COORDS,
    units='unitless',
    desc='List of the y-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

##### VERTICAL TAIL VALUES #####
# Metadata registration for DBF custom structural parameters
av.add_meta_data(
    Aircraft.VerticalTail.NUM_SPARS,
    units='unitless',
    desc='Number of tail spars',
    default_value=2,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.SPAR_THICKNESS,
    units='in',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.25,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.SPAR_DENSITY,
    units='lbm/in**3',
    desc='Material density of the spar',
    default_value=0.015,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.RIB_THICKNESS,
    units='in',
    desc='Thickness of a single rib',
    default_value=0.125,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.RIB_DENSITY,
    units='lbm/in**3',
    desc='Material density of the rib',
    default_value=0.012,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.SKIN_DENSITY,
    units='lbm/in**2',
    desc='Surface density of tail skin',
    default_value=0.02,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.AIRFOIL_X_COORDS,
    units='unitless',
    desc='List of the x-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.VerticalTail.AIRFOIL_Y_COORDS,
    units='unitless',
    desc='List of the y-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

##### FUSELAGE VALUES #####
# Metadata registration for DBF custom structural parameters
av.add_meta_data(
    Aircraft.Fuselage.NUM_SPARS,
    units='unitless',
    desc='Number of fuselage spars',
    default_value=2,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.SPAR_THICKNESS,
    units='in',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.25,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.SPAR_DENSITY,
    units='lbm/in**3',
    desc='Material density of the spar',
    default_value=0.015,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.NUM_RIBS,
    units='unitless',
    desc='Number of fuselage ribs',
    default_value=10,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.RIB_THICKNESS,
    units='in',
    desc='Thickness of a single rib',
    default_value=0.125,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.RIB_DENSITY,
    units='lbm/in**3',
    desc='Material density of the rib',
    default_value=0.012,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.SKIN_DENSITY,
    units='lbm/in**2',
    desc='Surface density of fuselage skin',
    default_value=0.02,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.FUSELAGE_HEIGHT,
    units='in',
    desc='Height of fuselage (assumed rectangular prism shape)',
    default_value=12,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Fuselage.FUSELAGE_WIDTH,
    units='in',
    desc='Width of fuselage (assumed rectangular prism shape)',
    default_value=12,
    meta_data=ExtendedMetaData,
)
