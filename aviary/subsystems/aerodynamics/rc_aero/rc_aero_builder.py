import openmdao.api as om

from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.subsystems.aerodynamics.rc_aero.rc_aero import TotalAircraftAero
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues


class RCAeroBuilder(SubsystemBuilder):
    def __init__(self, name='rc_aero_analysis'):
        super().__init__(name)

    def build(self, aviary_inputs: AviaryValues, aviary_outputs: AviaryValues):
        return TotalAircraftAero(aviary_inputs=aviary_inputs, num_nodes=1)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return TotalAircraftAero(
            aviary_inputs=aviary_inputs,
            num_nodes=num_nodes
        )

    def mission_inputs(self, **kwargs):
        mission_inputs = [
            Dynamic.Mission.ALTITUDE,
            Dynamic.Mission.VELOCITY,
        ]
        mission_inputs.extend(self.get_parameters(**kwargs).keys())
        return mission_inputs
    
    def mission_outputs(self, **kwargs):
        return [
            Dynamic.Vehicle.DRAG,
            Dynamic.Vehicle.LIFT,
            'alpha',
            'lifting_surface_CD',
            'avg_CL',
    ]

    def get_parameters(self, aviary_inputs=None, **kwargs):
        params = {}

        params[Aircraft.Wing.SPAN] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.Wing.ROOT_CHORD] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.Wing.SWEEP] = {
            'units': 'deg',
            'static_target': True
        }
        params[Aircraft.Wing.INCIDENCE] = {
            'units': 'deg',
            'static_target': True
        }
        params[Aircraft.HorizontalTail.SPAN] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.HorizontalTail.ROOT_CHORD] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.HorizontalTail.SWEEP] = {
            'units': 'deg',
            'static_target': True
        }
        params[Aircraft.Fuselage.MAX_HEIGHT] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.Fuselage.MAX_WIDTH] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.Fuselage.LENGTH] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.VerticalTail.SPAN] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.VerticalTail.ROOT_CHORD] = {
            'units': 'm',
            'static_target': True
        }
        return params

    def needs_mission_solver(self, aviary_inputs=None, subsystem_options=None, **kwargs):
        return False