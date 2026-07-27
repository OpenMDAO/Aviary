'''
The builder for aero external subsystem. 

inputs: altitude and velocity

outputs: lift, drag, alpha, avg_CL, lifting_surface_CD

parameters: geometry of wing, tail, fuselage; span, root_chord, sweep, etc

QUESTIONS:
    Should there be a PRE-MISSION for an aero external subsystem?

    Do we or do we not need needs_mission_solver,i.e. is there a solver that gets used?

    Is everything being called/used in a way that is up to date with 2026 Aviary?

    Where are the returned outputs being used as opposed to all of the other outputs 
    that warrants them being outputs and not the others?
'''

import openmdao.api as om

from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.subsystems.aerodynamics.UAV_Aero.aero_model import TotalAircraftAero
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues

class AeroBuilder(SubsystemBuilder):
    def __init__(self, name='UAV_Aero'):
        super().__init__(name)
#changed any def get input/output to mission inputs and outputs
    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None,
    ):
        return [
            Dynamic.Mission.ALTITUDE,
            Dynamic.Mission.VELOCITY,
            Dynamic.Vehicle.MASS,
            'aircraft:*',
        ]
    
    def mission_outputs(
    self,
    aviary_inputs=None,
    user_options=None,
    subsystem_options=None,
):
        return [
            Dynamic.Vehicle.LIFT,
            Dynamic.Vehicle.DRAG,
            Dynamic.Vehicle.DRAG_COEFFICIENT,
            'alpha',
            Dynamic.Vehicle.LIFT_COEFFICIENT,
            'lifting_surface_CD',
            'ht_area_ratio',
            'vt_area_ratio',
            'CD_fus',
            'CD_vtail',
            'CD_gear',
        ]
    
    def get_parameters(self, aviary_inputs=None, **kwargs):
        params = {}

        params[Aircraft.Wing.SPAN] = {
            'units': 'm',
            'static_target': True,
        }
        params[Aircraft.Wing.ROOT_CHORD] = {
            'units': 'm',
            'static_target': True,
        }
        params[Aircraft.Wing.SWEEP] = {
            'units': 'deg',
            'static_target': True,
        }
        params[Aircraft.Wing.INCIDENCE] = {
            'units': 'deg',
            'static_target': True
        }
        params[Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR] = {
        'units': 'unitless',
        'static_target': True,
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

    
    
    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return TotalAircraftAero(
            aviary_inputs=aviary_inputs,
            num_nodes=num_nodes
        )
    
    def needs_mission_solver(self, aviary_inputs=None, subsystem_options=None, **kwargs):
        return False      #changed from false to true