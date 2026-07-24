from aviary.subsystems.mass.UAV_mass.variable_info.mass_variables import Aircraft
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.subsystems.mass.UAV_mass.mass_premission import MassPremission
from aviary.subsystems.mass.UAV_mass.variable_info.mass_variable_metadata import ExtendedMetaData


class MassBuilder(SubsystemBuilder):
    _default_metadata = ExtendedMetaData
    
    """
    Builder for UAV mass models (wing, htail, vtail, fuselage, ...)
    """

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):

        subsystem_options = subsystem_options or {}
        return MassPremission(

            aviary_inputs = aviary_inputs,
            subsystem_options = subsystem_options,

        )
    
    def get_design_vars(self, aviary_inputs=None, user_options=None, subsystem_options=None, phase_info=None):
        DVs = {
            Aircraft.Wing.WETTED_AREA: {
                'units': 'm**2',
                'lower': 0.1,
                'upper': 5.0,
                'val': 2.0,  
            },
            Aircraft.Wing.SPAN: {
                'units': 'm',
                'lower': 0.1,
                'upper': 5.0,
                'val': 2.0,  
            },
            Aircraft.Wing.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 1.0,
                'val': 0.5,  
            },
            Aircraft.Fuselage.WETTED_AREA: {
                'units': 'm**2',
                'lower': 0.1,
                'upper': 5.0,
                'val': 2.0,  
            },
            Aircraft.Fuselage.LENGTH: {
                'units': 'm',
                'lower': 0.1,
                'upper': 5.0,
                'val': 1.0,  
            },
            Aircraft.Fuselage.AVG_HEIGHT: {
                'units': 'm',
                'lower': 0.1,
                'upper': 2.0,
                'val': 0.5,  
            },
            Aircraft.Fuselage.AVG_WIDTH: {
                'units': 'm',
                'lower': 0.1,
                'upper': 2.0,
                'val': 0.5,  
            },
            Aircraft.HorizontalTail.SPAN: {
                'units': 'm',
                'lower': 0.1,
                'upper': 5.0,
                'val': 2.0,  
            },
            Aircraft.HorizontalTail.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 5.0,
                'val': 1.0,  
            },
            Aircraft.HorizontalTail.WETTED_AREA: {
                'units': 'm**2',
                'lower': 0.1,
                'upper': 5.0,
                'val': 2.0,  
            },
            Aircraft.VerticalTail.SPAN: {
                'units': 'm',
                'lower': 0.1,
                'upper': 5.0,
                'val': 1.0,  
            },
            Aircraft.VerticalTail.ROOT_CHORD: {
                'units': 'm',
                'lower': 0.1,
                'upper': 1.0,
                'val': 0.5,  
            },
            Aircraft.VerticalTail.WETTED_AREA: {
                'units': 'm**2',
                'lower': 0.1,
                'upper': 5.0,
                'val': 2.0,  
            },
        }
        return DVs
    
    def get_inputs(self):

        return [

            Aircraft.Wing.SPAN,
            Aircraft.Wing.ROOT_CHORD,
            Aircraft.Wing.WETTED_AREA,
            Aircraft.Fuselage.LENGTH,
            Aircraft.Fuselage.AVG_HEIGHT,
            Aircraft.Fuselage.AVG_WIDTH,
            Aircraft.Fuselage.WETTED_AREA,
            Aircraft.HorizontalTail.SPAN,
            Aircraft.HorizontalTail.ROOT_CHORD,
            Aircraft.HorizontalTail.WETTED_AREA,
            Aircraft.VerticalTail.SPAN,
            Aircraft.VerticalTail.ROOT_CHORD,
            Aircraft.VerticalTail.WETTED_AREA,

        ]
    
    def get_outputs(self):

        return [

            Aircraft.Wing.MASS,
            Aircraft.HorizontalTail.MASS,
            Aircraft.VerticalTail.MASS,
            Aircraft.Fuselage.MASS,
            Aircraft.Design.STRUCTURE_MASS,

        ]

   