
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft


def check_fold_location_definition(inputs, options: AviaryValues):
    choose_fold_location = options.get_val(
        Aircraft.Wing.CHOOSE_FOLD_LOCATION, units='unitless')
    has_strut = options.get_val(Aircraft.Wing.HAS_STRUT, units='unitless')
    if not choose_fold_location and not has_strut:
        raise RuntimeError(
            "The option CHOOSE_FOLD_LOCATION can only be False if the option HAS_STRUT is True.")

# Possible TODO
# Aircraft.Design.ULF_CALCULATED_FROM_MANEUVER - Aircraft.Design.PART25_STRUCTURAL_CATEGORY
# Aircraft.Engine.FUSELAGE_MOUNTED - Aircraft.Engine.WING_LOCATIONS
# Aircraft.Engine.NUM_ENGINES - Aircraft.Engine.NUM_FUSELAGE_ENGINES - Aircraft.Engine.NUM_WING_ENGINES
# Aircraft.Propulsion.TOTAL_NUM_ENGINES - Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES - Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES
# Aircraft.Engine.TYPE - Aircraft.Engine.HAS_PROPELLERS
# Aircraft.Design.COMPUTE_TAIL_VOLUME_COEFFS
# Aircraft.Engine.REFERENCE_WEIGHT
# Aircraft.Fuselage.PROVIDE_SURFACE_AREA - Aircraft.Fuselage.WETTED_AREA_FACTOR
# Mission.Taxi.MACH - pycycle
# Aircraft.HorizontalTail.AREA
# Aircraft.VerticalTail.AREA
# Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED
# Aircraft.Wing.FOLD_LOCATION_IS_DIMENSIONAL
# Aircraft.Wing.HAS_FOLD
