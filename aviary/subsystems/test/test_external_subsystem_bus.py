"""
    Test external subsystem bus API.
"""
from copy import deepcopy
import pkg_resources
import unittest

import numpy as np

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

import aviary.api as av
from aviary.interface.default_phase_info.height_energy import phase_info as ph_in
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase


class PreMissionComp(om.ExplicitComponent):

    def setup(self):
        self.add_output('for_climb', np.ones((2, 1)), units='ft')
        self.add_output('for_cruise', np.ones((2, 1)), units='ft')
        self.add_output('for_descent', np.ones((2, 1)), units='ft')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['for_climb'] = np.array([[3.1], [1.7]])
        outputs['for_cruise'] = np.array([[1.2], [4.1]])
        outputs['for_descent'] = np.array([[3.], [8.]])


class MissionComp(om.ExplicitComponent):

    def initialize(self):
        self.options.declare('shape', default=(1, ))

    def setup(self):
        shape = self.options['shape']
        self.add_input('xx', shape=shape, units='ft')
        self.add_output('yy', shape=shape, units='ft')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        outputs['yy'] = 2.0 * inputs['xx']


class CustomBuilder(SubsystemBuilderBase):

    def build_pre_mission(self, aviary_inputs):
        return PreMissionComp()

    def build_mission(self, num_nodes, aviary_inputs):
        sub_group = om.Group()
        sub_group.add_subsystem('electric', MissionComp(shape=(2, 1)),
                                promotes_inputs=['*'],
                                promotes_outputs=['*'],
                                )
        return sub_group

    def get_bus_variables(self):
        vars_to_connect = {
            "test.for_climb": {
                "mission_name": ['test.xx'],
                "units": 'ft',
                "shape": (2, 1),
                "phases": ['climb']
            },
            "test.for_cruise": {
                "mission_name": ['test.xx'],
                "units": 'ft',
                "shape": (2, 1),
                "phases": ['cruise']
            },
            "test.for_descent": {
                "mission_name": ['test.xx'],
                "units": 'ft',
                "shape": (2, 1),
                "phases": ['descent']
            },
        }

        return vars_to_connect


class TestExternalSubsystemBus(unittest.TestCase):

    def test_external_subsystem_bus(self):
        phase_info = deepcopy(ph_in)
        phase_info['pre_mission']['external_subsystems'] = [CustomBuilder(name='test')]
        phase_info['climb']['external_subsystems'] = [CustomBuilder(name='test')]
        phase_info['cruise']['external_subsystems'] = [CustomBuilder(name='test')]
        phase_info['descent']['external_subsystems'] = [CustomBuilder(name='test')]

        prob = AviaryProblem()

        csv_path = pkg_resources.resource_filename(
            "aviary", "models/test_aircraft/aircraft_for_bench_FwFm.csv")
        prob.load_inputs(csv_path, phase_info)
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.setup()
        prob.set_initial_guesses()

        # Just run once to pass data.
        prob.run_model()

        # Each phase should have a different value passed from pre using the bus.
        assert_near_equal(
            prob.model.get_val('traj.climb.rhs_all.test.yy'),
            2.0 * np.array([[3.1], [1.7]]),
        )
        assert_near_equal(
            prob.model.get_val('traj.cruise.rhs_all.test.yy'),
            2.0 * np.array([[1.2], [4.1]]),
        )
        assert_near_equal(
            prob.model.get_val('traj.descent.rhs_all.test.yy'),
            2.0 * np.array([[3.], [8.]]),
        )


if __name__ == '__main__':
    unittest.main()
