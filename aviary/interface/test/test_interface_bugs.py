import unittest
from copy import deepcopy

import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.models.test_aircraft.GwFm_phase_info import phase_info as ph_in
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.functions import get_aviary_resource_path
from aviary.variable_info.variables import Aircraft


class WingWeightSubsys(om.ExplicitComponent):
    def setup(self):
        self.add_input(Aircraft.Engine.MASS, 1.0, units='lbm')
        self.add_output(Aircraft.Canard.ASPECT_RATIO, 1.0, units='unitless')
        self.add_output('Tail', 1.0, units='unitless')

        self.declare_partials(Aircraft.Canard.ASPECT_RATIO, Aircraft.Engine.MASS, val=2.0)
        self.declare_partials('Tail', Aircraft.Engine.MASS, val=0.7)

    def compute(self, inputs, outputs):
        outputs[Aircraft.Canard.ASPECT_RATIO] = 2.0 * inputs[Aircraft.Engine.MASS]
        outputs['Tail'] = 0.7 * inputs[Aircraft.Engine.MASS]


class WingWeightBuilder(SubsystemBuilderBase):
    """Prototype of a subsystem that overrides an aviary internally computed var."""

    def __init__(self, name='wing_weight'):
        super().__init__(name)

    def build_post_mission(self, aviary_inputs, phase_info, phase_mission_bus_lengths):
        """
        Build an OpenMDAO system for the pre-mission computations of the subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.System
            An OpenMDAO system containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        """
        wing_group = om.Group()
        wing_group.add_subsystem(
            'aerostructures',
            WingWeightSubsys(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[
                Aircraft.Canard.ASPECT_RATIO,
                ('Tail', Aircraft.Canard.WETTED_AREA_SCALER),
            ],
        )
        return wing_group


@use_tempdirs
class PreMissionGroupTest(unittest.TestCase):
    def test_post_mission_promotion(self):
        phase_info = deepcopy(ph_in)
        phase_info['post_mission'] = {}
        phase_info['post_mission']['include_landing'] = False
        phase_info['post_mission']['external_subsystems'] = [
            WingWeightBuilder(name='wing_external')
        ]

        prob = AviaryProblem()

        csv_path = get_aviary_resource_path('models/test_aircraft/aircraft_for_bench_GwFm.csv')
        prob.load_inputs(csv_path, phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('SLSQP', verbosity=0)

        prob.add_design_variables()

        prob.add_objective(objective_type='mass', ref=-1e5)

        prob.setup()

    def test_serial_phase_group(self):
        phase_info = deepcopy(ph_in)
        phase_info['post_mission'] = {}
        phase_info['post_mission']['include_landing'] = False
        phase_info['post_mission']['external_subsystems'] = [
            WingWeightBuilder(name='wing_external')
        ]

        prob = AviaryProblem()

        csv_path = get_aviary_resource_path('models/test_aircraft/aircraft_for_bench_GwFm.csv')
        prob.load_inputs(csv_path, phase_info)

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases(parallel_phases=False)

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('SLSQP', verbosity=0)

        prob.add_design_variables()

        prob.add_objective(objective_type='mass', ref=-1e5)

        prob.setup()

        assert not isinstance(prob.traj.phases, om.ParallelGroup)


if __name__ == '__main__':
    unittest.main()
