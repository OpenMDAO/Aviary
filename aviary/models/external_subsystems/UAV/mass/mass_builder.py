from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft
from aviary.models.external_subsystems.UAV.mass.model.mass_premission import MassPremission
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variable_meta_data import (
    ExtendedMetaData,
)
from aviary.subsystems.subsystem_builder import SubsystemBuilder


class MassBuilder(SubsystemBuilder):
    _default_metadata = ExtendedMetaData

    """
    Builder for UAV mass models (wing, htail, vtail, fuselage, ...)
    """

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        subsystem_options = subsystem_options or {}
        return MassPremission(
            aviary_inputs=aviary_inputs,
            subsystem_options=subsystem_options,
        )

    def get_design_vars(
        self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None
    ):
        DVs = {
            # WETTED_AREA is no longer a design variable for any surface: each mass component
            # now computes its wetted area as span * root_chord (see wing.py / horizontaltail.py /
            # verticaltail.py) -- the same areas the aero forms in UAV_Aero/aero_model.py
            # (WingTailAreaRatios) -- so span & chord drive both lift/drag and skin/sheeting mass.
            # Starting values live in the aircraft CSV, not here: OpenMDAO's
            Aircraft.Wing.SPAN: {
                'units': 'm',
                'lower': 1.0,
                'upper': 3,
            },
            Aircraft.Wing.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 2.0,
            },
            Aircraft.HorizontalTail.SPAN: {
                'units': 'm',
                'lower': 0.1,
                'upper': 2.0,
            },
            Aircraft.HorizontalTail.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 1.0,
            },
            Aircraft.VerticalTail.SPAN: {
                'units': 'm',
                'lower': 0.05,
                'upper': 0.5,
            },
            Aircraft.VerticalTail.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 1.0,
            },
        }
        return DVs

    def get_inputs(self):
        return [
            Aircraft.Wing.SPAN,
            Aircraft.Wing.ROOT_CHORD,
            Aircraft.Fuselage.LENGTH,
            Aircraft.Fuselage.AVG_HEIGHT,
            Aircraft.Fuselage.AVG_WIDTH,
            Aircraft.Fuselage.WETTED_AREA,
            Aircraft.HorizontalTail.SPAN,
            Aircraft.HorizontalTail.ROOT_CHORD,
            Aircraft.VerticalTail.SPAN,
            Aircraft.VerticalTail.ROOT_CHORD,
        ]

    def get_outputs(self):
        return [
            Aircraft.Wing.MASS,
            Aircraft.HorizontalTail.MASS,
            Aircraft.VerticalTail.MASS,
            Aircraft.Fuselage.MASS,
            Aircraft.Design.STRUCTURE_MASS,
        ]
