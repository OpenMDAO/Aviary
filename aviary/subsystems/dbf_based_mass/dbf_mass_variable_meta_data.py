import aviary.api as av
from aviary.subsystems.dbf_based_mass.dbf_mass_variables import Aircraft

ExtendedMetaData = av.CoreMetaData

##### WING VALUES #####
# Metadata registration for DBF custom structural parameters
av.add_meta_data(
    Aircraft.DBFWing.NUM_SPARS,
    units='unitless',
    desc='Number of wing spars',
    default_value=2,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.SPAR_THICKNESS,
    units='in',
    desc='Diameter/thickness of a single spar (assumed cylindrical)',
    default_value=0.25,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.SPAR_DENSITY,
    units='lbm/in**3',
    desc='Material density of the spar',
    default_value=0.015,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.NUM_RIBS,
    units='unitless',
    desc='Number of wing ribs',
    default_value=10,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.RIB_THICKNESS,
    units='in',
    desc='Thickness of a single rib',
    default_value=0.125,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.RIB_DENSITY,
    units='lbm/in**3',
    desc='Material density of the rib',
    default_value=0.012,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.CROSS_SECTIONAL_AREA,
    units='in**2',
    desc='Cross-sectional area of a rib',
    default_value=12.0,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.SKIN_DENSITY,
    units='lbm/in**2',
    desc='Surface density of wing skin',
    default_value=0.02,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.AIRFOIL_X_COORDS,
    units='unitless',
    desc='List of the x-coords of an airfoil shape',
    default_value=[],
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.DBFWing.AIRFOIL_Y_COORDS,
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
    Aircraft.HorizontalTail.CROSS_SECTIONAL_AREA,
    units='in**2',
    desc='Cross-sectional area of a rib',
    default_value=12.0,
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
    Aircraft.VerticalTail.CROSS_SECTIONAL_AREA,
    units='in**2',
    desc='Cross-sectional area of a rib',
    default_value=12.0,
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
