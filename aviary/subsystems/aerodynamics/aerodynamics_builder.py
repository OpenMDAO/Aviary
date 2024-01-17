"""
Define subsystem builder for Aviary core aerodynamics.

Classes
-------
AerodynamicsBuilderBase : the interface for an aerodynamics subsystem builder.

CoreAerodynamicsBuilder : the interface for Aviary's core aerodynamics subsystem builder
"""
import openmdao.api as om

from aviary.variable_info.variables import Aircraft, Mission, Dynamic

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.aerodynamics.flops_based.aero_report import AeroReport
from aviary.subsystems.aerodynamics.flops_based.design import Design
from aviary.subsystems.aerodynamics.gasp_based.premission_aero import PreMissionAero
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import CruiseAero
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import LowSpeedAero
from aviary.subsystems.aerodynamics.gasp_based.table_based import CruiseAero as TabularCruiseAero
from aviary.subsystems.aerodynamics.gasp_based.table_based import LowSpeedAero as TabularLowSpeedAero
from aviary.subsystems.aerodynamics.flops_based.computed_aero_group import \
    ComputedAeroGroup
from aviary.subsystems.aerodynamics.flops_based.takeoff_aero_group import \
    TakeoffAeroGroup
from aviary.subsystems.aerodynamics.flops_based.solved_alpha_group import \
    SolvedAlphaGroup
from aviary.subsystems.aerodynamics.flops_based.tabular_aero_group import \
    TabularAeroGroup
from aviary.variable_info.enums import LegacyCode


GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS

_default_name = 'aerodynamics'


class AerodynamicsBuilderBase(SubsystemBuilderBase):
    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CoreAerodynamicsBuilder(AerodynamicsBuilderBase):
    def __init__(self, name=None, meta_data=None, code_origin=None):
        if name is None:
            name = 'core_aerodynamics'

        if code_origin not in (FLOPS, GASP):
            raise ValueError('Code origin is not one of the following: (FLOPS, GASP)')

        self.code_origin = code_origin

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs):
        code_origin = self.code_origin

        if code_origin is GASP:
            aero_group = PreMissionAero(aviary_options=aviary_inputs)

        elif code_origin is FLOPS:
            aero_group = om.Group()
            aero_group.add_subsystem(
                'design', Design(aviary_options=aviary_inputs),
                promotes_inputs=['*'],
                promotes_outputs=['*'])

            aero_group.add_subsystem(
                'aero_report', AeroReport(aviary_options=aviary_inputs),
                promotes_inputs=['*'],
                promotes_outputs=['*'])

        return aero_group

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None
        if self.code_origin is FLOPS:
            if method is None:
                aero_group = ComputedAeroGroup(num_nodes=num_nodes,
                                               aviary_options=aviary_inputs)

            elif method == 'computed':
                aero_group = ComputedAeroGroup(num_nodes=num_nodes,
                                               aviary_options=aviary_inputs,
                                               **kwargs)

            elif method == 'low_speed':
                aero_group = TakeoffAeroGroup(num_nodes=num_nodes,
                                              aviary_options=aviary_inputs,
                                              **kwargs)

            # TODO solved alpha belongs in the GASP side, rolled into tabular aero
            #      It is currently only here because it is not possible to define
            #      per-subsystem code origins in AviaryProblem yet
            elif method == 'solved_alpha':
                aero_group = SolvedAlphaGroup(num_nodes=num_nodes,
                                              aero_data=kwargs.pop('aero_data'),
                                              **kwargs)

            elif method == 'tabular':
                aero_group = TabularAeroGroup(num_nodes=num_nodes,
                                              CD0_data=kwargs.pop('CD0_data'),
                                              CDI_data=kwargs.pop('CDI_data'),
                                              **kwargs)

            else:
                raise ValueError('FLOPS-based aero method is not one of the following: '
                                 '(computed, low_speed, solved_alpha, tabular)')

        elif self.code_origin is GASP:
            if method is None:
                aero_group = CruiseAero(num_nodes=num_nodes)

            elif method == 'cruise':
                if 'aero_data' in kwargs:
                    aero_group = TabularCruiseAero(num_nodes=num_nodes,
                                                   aero_data=kwargs.pop('aero_data'),
                                                   **kwargs)
                else:
                    aero_group = CruiseAero(num_nodes=num_nodes,
                                            **kwargs)

            elif method == 'low_speed':
                if any(key in kwargs for key in ['free_aero_data',
                                                 'free_flaps_data',
                                                 'free_ground_data']) in kwargs:
                    aero_group = TabularLowSpeedAero(num_nodes=num_nodes,
                                                     free_aero_data=kwargs.pop(
                                                         'free_aero_data'),
                                                     free_flaps_data=kwargs.pop(
                                                         'free_flaps_data'),
                                                     free_ground_data=kwargs.pop(
                                                         'free_ground_data'),
                                                     **kwargs)

                else:
                    aero_group = LowSpeedAero(num_nodes=num_nodes,
                                              **kwargs)

            else:
                raise ValueError('GASP-based aero method is not one of the following: '
                                 '(cruise, low_speed)')

        return aero_group

    def mission_inputs(self, **kwargs):
        method = kwargs['method']
        promotes = ['*']

        if self.code_origin is FLOPS:
            if method == 'computed':
                promotes = [Dynamic.Mission.STATIC_PRESSURE,
                            Dynamic.Mission.MACH,
                            Dynamic.Mission.TEMPERATURE,
                            Dynamic.Mission.MASS,
                            'aircraft:*', 'mission:*']

            elif method == 'solved_alpha':
                promotes = [Dynamic.Mission.ALTITUDE,
                            Dynamic.Mission.MACH,
                            Dynamic.Mission.MASS,
                            Dynamic.Mission.STATIC_PRESSURE,
                            'aircraft:*']

            elif method == 'low_speed':
                promotes = ['angle_of_attack',
                            Dynamic.Mission.ALTITUDE,
                            Dynamic.Mission.FLIGHT_PATH_ANGLE,
                            Mission.Takeoff.DRAG_COEFFICIENT_MIN,
                            Aircraft.Wing.ASPECT_RATIO,
                            Aircraft.Wing.HEIGHT,
                            Aircraft.Wing.SPAN,
                            Dynamic.Mission.DYNAMIC_PRESSURE,
                            Aircraft.Wing.AREA]

            elif method == 'tabular':
                promotes = [Dynamic.Mission.ALTITUDE,
                            Dynamic.Mission.MACH,
                            Dynamic.Mission.MASS,
                            Dynamic.Mission.VELOCITY,
                            Dynamic.Mission.DENSITY,
                            'aircraft:*']

            else:
                raise ValueError('FLOPS-based aero method is not one of the following: '
                                 '(computed, low_speed, solved_alpha, tabular)')

        elif self.code_origin is GASP:
            if method == 'low_speed':
                promotes = ['*',
                            ("airport_alt", Mission.Takeoff.AIRPORT_ALTITUDE),
                            ("CL_max_flaps", Mission.Takeoff.LIFT_COEFFICIENT_MAX),
                            ("dCL_flaps_model", Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT),
                            ("dCD_flaps_model", Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT),
                            ("flap_defl", Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF)]

            elif method == 'cruise':
                if 'output_alpha' in kwargs:
                    if kwargs['output_alpha']:
                        promotes = ['*', ("lift_req", "weight")]

            else:
                raise ValueError('GASP-based aero method is not one of the following: '
                                 '(low_speed, cruise)')

        return promotes

    def mission_outputs(self, **kwargs):
        method = kwargs['method']
        promotes = ['*']

        if self.code_origin is FLOPS:
            promotes = [Dynamic.Mission.DRAG]
            if method == 'low_speed':
                promotes.append(Dynamic.Mission.LIFT)

        elif self.code_origin is GASP:
            if method == 'low_speed':
                promotes = [Dynamic.Mission.DRAG,
                            Dynamic.Mission.LIFT,
                            'CL', 'CD', 'flap_factor', 'gear_factor']

            elif method == 'cruise':
                if 'output_alpha' in kwargs:
                    if kwargs['output_alpha']:
                        promotes = [Dynamic.Mission.DRAG,
                                    Dynamic.Mission.LIFT,
                                    'alpha']
                else:
                    promotes = [Dynamic.Mission.DRAG,
                                Dynamic.Mission.LIFT,
                                'CL_max']

            else:
                raise ValueError('GASP-based aero method is not one of the following: '
                                 '(low_speed, cruise)')

        return promotes

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core aerodynamics analysis

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        if self.code_origin is FLOPS:
            # FLOPS aero report goes here
            return
        elif self.code_origin is GASP:
            # GASP aero report goes here
            return
