from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.subsystems.mass.mass_builder import CoreMassBuilder
from aviary.subsystems.propulsion.propulsion_builder import CorePropulsionBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData


def get_default_premission_subsystems(legacy_code, engines=None):
    """
    Get default premission subsystems propulsion, geometry, aerodynamics, and mass
    in this order.

    Arguments:
    ----------
    legacy_code : str, LegacyCode
        either FLOPS or GASP LegacyCode Enums, or their strings equivalents ('FLOPS', 'GASP')
    engine : <list of EngineDecks>
        List of EngineDecks
    """
    legacy_code = LegacyCode(legacy_code)
    prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engine_models=engines)
    mass = CoreMassBuilder('core_mass', BaseMetaData, legacy_code)
    aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, legacy_code)
    geom = CoreGeometryBuilder('core_geometry', BaseMetaData, legacy_code)

    return [prop, geom, aero, mass]


def get_default_mission_subsystems(legacy_code, engines=None):
    """
    Get default mission subsystems aerodynamics and propulsion in this order.

    Arguments:
    ----------
    legacy_code : str, LegacyCode
        either FLOPS or GASP LegacyCode Enums, or their strings equivalents ('FLOPS', 'GASP')
    engine : <list of EngineDecks>
        List of EngineDecks
    """
    legacy_code = LegacyCode(legacy_code)
    prop = CorePropulsionBuilder('core_propulsion', BaseMetaData, engine_models=engines)
    aero = CoreAerodynamicsBuilder('core_aerodynamics', BaseMetaData, legacy_code)

    return [aero, prop]


def get_geom_and_mass_subsystems(legacy_code):
    """
    Get geometry and mass premission subsystems in this order.

    Arguments:
    ----------
    legacy_code : str, LegacyCode
        either FLOPS or GASP LegacyCode Enums, or their strings equivalents ('FLOPS', 'GASP')
    """
    legacy_code = LegacyCode(legacy_code)
    mass = CoreMassBuilder('core_mass', BaseMetaData, legacy_code)
    geom = CoreGeometryBuilder('core_geometry', BaseMetaData, legacy_code)

    return [geom, mass]
