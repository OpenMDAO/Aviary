import openmdao.api as om

from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.models.external_subsystems.UAV.aerodynamics.model.aero_model import TotalAircraftAero
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues


class AeroBuilder(SubsystemBuilder):
    def __init__(self, name='UAV_Aero'):
        super().__init__(name)

    # changed any def get input/output to mission inputs and outputs
    def mission_inputs(
        self,
        aviary_inputs=None,
        user_options=None,
        subsystem_options=None,
    ):
        return [
            Dynamic.Mission.ALTITUDE,
            Dynamic.Mission.VELOCITY,
            Dynamic.Vehicle.MASS,
            'aircraft:*',
            'alpha',
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
            'lift_balance_residual',
            Dynamic.Vehicle.LIFT_COEFFICIENT,
            'lifting_surface_CD',
            'ht_area_ratio',
            'vt_area_ratio',
            'CD_fus',
            'CD_vtail',
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
        params[Aircraft.Wing.INCIDENCE] = {'units': 'deg', 'static_target': True}
        params[Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR] = {
            'units': 'unitless',
            'static_target': True,
        }
        params[Aircraft.HorizontalTail.SPAN] = {'units': 'm', 'static_target': True}
        params[Aircraft.HorizontalTail.ROOT_CHORD] = {'units': 'm', 'static_target': True}
        params[Aircraft.HorizontalTail.SWEEP] = {'units': 'deg', 'static_target': True}
        params[Aircraft.Fuselage.MAX_HEIGHT] = {'units': 'm', 'static_target': True}
        params[Aircraft.Fuselage.MAX_WIDTH] = {'units': 'm', 'static_target': True}
        params[Aircraft.Fuselage.LENGTH] = {'units': 'm', 'static_target': True}
        params[Aircraft.VerticalTail.SPAN] = {'units': 'm', 'static_target': True}
        params[Aircraft.VerticalTail.ROOT_CHORD] = {'units': 'm', 'static_target': True}
        return params

    def get_controls(self, aviary_inputs=None, **kwargs):
        controls = {
            'alpha': {
                'units': 'deg',
                'targets': 'alpha',
                'opt': True,
                'val': 0.0,
                'lower': -10.0,
                'upper': 10.0,
                'ref': 5.0,
                # external-subsystem controls bypass Aviary's add_control, so
                # continuity defects otherwise default to ref=1 (row was 1.4e2)
                'continuity_ref': 5.0,
                'rate_continuity_ref': 5.0,
            },
        }

        return controls

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return TotalAircraftAero(aviary_inputs=aviary_inputs, num_nodes=num_nodes)

    def needs_mission_solver(self, aviary_inputs=None, subsystem_options=None, **kwargs):
        return False
