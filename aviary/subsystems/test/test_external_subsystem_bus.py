"""Test external subsystem bus API."""

import unittest
from copy import deepcopy

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

import aviary.api as av
from aviary.interface.default_phase_info.height_energy import phase_info as ph_in
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Dynamic


ExtendedMetaData = av.CoreMetaData


av.add_meta_data(
    'the_shape_for_the_thing_dim0',
    units='unitless',
    desc='length for the first dimension PreMissionComp outputs',
    default_value=2,
    types=int,
    option=True,
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    'the_shape_for_the_thing_dim1',
    units='unitless',
    desc='length for the second dimension PreMissionComp outputs',
    default_value=3,
    types=int,
    option=True,
    meta_data=ExtendedMetaData,
)


class PreMissionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', default=(2, 3))

    def setup(self):
        shape = self.options['shape']
        self.add_output('for_climb', np.ones(shape), units='ft')
        self.add_output('for_cruise', np.ones(shape), units='ft')
        self.add_output('for_descent', np.ones(shape), units='ft')

    def compute(self, inputs, outputs):
        shape = self.options['shape']
        outputs['for_climb'] = np.random.random(shape)
        outputs['for_cruise'] = np.random.random(shape)
        outputs['for_descent'] = np.random.random(shape)


class MissionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('shape', default=1)
        self.options.declare('num_nodes', default=1)

    def setup(self):
        shape = self.options['shape']
        num_nodes = self.options['num_nodes']
        self.add_input('xx', shape=shape, units='ft')
        self.add_output('yy', shape=num_nodes, units='ft')
        self.add_output('zz', shape=num_nodes, units='ft')

    def compute(self, inputs, outputs):
        num_nodes = self.options['num_nodes']
        outputs['yy'] = 2.0 * np.sum(inputs['xx']) * range(num_nodes)
        outputs['zz'] = 3.0 * np.sum(inputs['xx']) * range(num_nodes)


class PostMissionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('do_the_zz_thing', types=bool, default=False)
        self.options.declare('shape', default=(2, 3))
        self.options.declare('num_nodes', default=1)

    def setup(self):
        shape = self.options['shape']
        num_nodes = self.options['num_nodes']
        self.add_input('xx', shape=shape, units='ft')
        if self.options['do_the_zz_thing']:
            self.add_input('zz', shape=num_nodes, units='ft')
            self.add_input('velocity', shape=num_nodes, units='ft/s')
        self.add_output('zzz', shape=1, units='ft')

    def compute(self, inputs, outputs):
        outputs['zzz'] = np.sum(inputs['xx'])
        if self.options['do_the_zz_thing']:
            outputs['zzz'] *= np.sum(inputs['zz'] * inputs['velocity'])


class CustomBuilder(SubsystemBuilderBase):
    def __init__(self, name, mangle_names=False):
        self.mangle_names = mangle_names
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        shape = (
            aviary_inputs.get_val('the_shape_for_the_thing_dim0'),
            aviary_inputs.get_val('the_shape_for_the_thing_dim1'),
        )
        if self.mangle_names:
            sys = om.Group()
            name = self.name
            sys.add_subsystem(
                'foo',
                PreMissionComp(shape=shape),
                promotes_outputs=[
                    ('for_climb', f'{name}_for_climb'),
                    ('for_cruise', f'{name}_for_cruise'),
                    ('for_descent', f'{name}_for_descent'),
                ],
            )

        else:
            sys = PreMissionComp(shape=shape)
        return sys

    def build_mission(self, num_nodes, aviary_inputs):
        sub_group = om.Group()
        shape = (
            aviary_inputs.get_val('the_shape_for_the_thing_dim0'),
            aviary_inputs.get_val('the_shape_for_the_thing_dim1'),
        )
        comp = MissionComp(shape=shape, num_nodes=num_nodes)
        if self.mangle_names:
            name = self.name
            sub_group.add_subsystem(
                'electric',
                comp,
                promotes_inputs=[('xx', f'{name}_xx')],
                promotes_outputs=[('yy', f'{name}_yy'), ('zz', f'{name}_zz')],
            )
        else:
            sub_group.add_subsystem(
                'electric',
                comp,
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        return sub_group

    def get_pre_mission_bus_variables(self, aviary_inputs):
        shape = (
            aviary_inputs.get_val('the_shape_for_the_thing_dim0'),
            aviary_inputs.get_val('the_shape_for_the_thing_dim1'),
        )
        name = self.name
        if self.mangle_names:
            vars_to_connect = {
                f'{name}.{name}_for_climb': {
                    'mission_name': [f'{name}.{name}_xx'],
                    'post_mission_name': f'{name}.climb_{name}_xx',
                    'units': 'ft',
                    'shape': shape,
                    'phases': ['climb'],
                },
                f'{name}.{name}_for_cruise': {
                    'mission_name': [f'{name}.{name}_xx'],
                    'post_mission_name': f'{name}.cruise_{name}_xx',
                    'units': 'ft',
                    'shape': shape,
                    'phases': ['cruise'],
                },
                f'{name}.{name}_for_descent': {
                    'mission_name': [f'{name}.{name}_xx'],
                    'post_mission_name': [f'{name}.descent_{name}_xx'],
                    'units': 'ft',
                    'shape': shape,
                    'phases': ['descent'],
                },
            }
        else:
            vars_to_connect = {
                f'{name}.for_climb': {
                    'mission_name': [f'{name}.xx'],
                    'post_mission_name': f'{name}.climb_xx',
                    'units': 'ft',
                    'shape': shape,
                    'phases': ['climb'],
                },
                f'{name}.for_cruise': {
                    'mission_name': [f'{name}.xx'],
                    'post_mission_name': f'{name}.cruise_xx',
                    'units': 'ft',
                    'shape': shape,
                    'phases': ['cruise'],
                },
                f'{name}.for_descent': {
                    'mission_name': [f'{name}.xx'],
                    'post_mission_name': [f'{name}.descent_xx'],
                    'units': 'ft',
                    'shape': shape,
                    'phases': ['descent'],
                },
            }
        return vars_to_connect

    def get_post_mission_bus_variables(self, aviary_inputs, phase_info):
        name = self.name
        out = {}
        for phase_name, phase_data in phase_info.items():
            phase_d = {}
            if phase_data.get('do_the_zz_thing', False):
                if self.mangle_names:
                    phase_d[f'{name}.{name}_zz'] = f'{name}.{phase_name}_{name}_zz'
                    phase_d[Dynamic.Mission.VELOCITY] = f'{name}.{phase_name}_{name}_velocity'
                else:
                    phase_d[f'{name}.zz'] = f'{name}.{phase_name}_zz'
                    phase_d[Dynamic.Mission.VELOCITY] = f'{name}.{phase_name}_velocity'
            out[phase_name] = phase_d
        return out

    def build_post_mission(self, aviary_inputs, phase_info, phase_mission_bus_lengths, **kwargs):
        shape = (
            aviary_inputs.get_val('the_shape_for_the_thing_dim0'),
            aviary_inputs.get_val('the_shape_for_the_thing_dim1'),
        )
        group = om.Group()
        name = self.name
        for phase_name, phase_data in phase_info.items():
            do_the_zz_thing = phase_data.get('do_the_zz_thing', False)
            num_nodes = phase_mission_bus_lengths[phase_name]
            comp = PostMissionComp(
                num_nodes=num_nodes, shape=shape, do_the_zz_thing=do_the_zz_thing
            )
            if self.mangle_names:
                pi = [('xx', f'{phase_name}_{name}_xx')]
                if do_the_zz_thing:
                    pi.append(('zz', f'{phase_name}_{name}_zz'))
                    pi.append(('velocity', f'{phase_name}_{name}_velocity'))
                po = [('zzz', f'{phase_name}_{name}_zzz')]
            else:
                pi = [('xx', f'{phase_name}_xx')]
                if do_the_zz_thing:
                    pi.append(('zz', f'{phase_name}_zz'))
                    pi.append(('velocity', f'{phase_name}_velocity'))
                po = [('zzz', f'{phase_name}_zzz')]
            group.add_subsystem(
                f'{phase_name}_post_mission', comp, promotes_inputs=pi, promotes_outputs=po
            )
        return group


@use_tempdirs
class TestExternalSubsystemBus(unittest.TestCase):
    def test_external_subsystem_bus(self):
        phase_info = deepcopy(ph_in)
        # Adding two `CustomBuilder` external subsystems will test that we can request `Dynamic.Mission.VELOCITY` be a mission bus variable twice.
        phase_info['pre_mission']['external_subsystems'] = [
            CustomBuilder(name='test'),
            CustomBuilder(name='test2', mangle_names=True),
        ]
        phase_info['climb']['external_subsystems'] = [
            CustomBuilder(name='test'),
            CustomBuilder(name='test2', mangle_names=True),
        ]
        phase_info['cruise']['external_subsystems'] = [
            CustomBuilder(name='test'),
            CustomBuilder(name='test2', mangle_names=True),
        ]
        phase_info['descent']['external_subsystems'] = [
            CustomBuilder(name='test'),
            CustomBuilder(name='test2', mangle_names=True),
        ]
        phase_info['post_mission']['external_subsystems'] = [
            CustomBuilder(name='test'),
            CustomBuilder(name='test2', mangle_names=True),
        ]

        phase_info['climb']['do_the_zz_thing'] = True
        phase_info['descent']['do_the_zz_thing'] = False

        prob = AviaryProblem()

        csv_path = 'models/test_aircraft/aircraft_for_bench_FwFm.csv'
        prob.load_inputs(csv_path, phase_info)
        prob.aviary_inputs.set_val('the_shape_for_the_thing_dim0', 3, meta_data=ExtendedMetaData)
        prob.aviary_inputs.set_val('the_shape_for_the_thing_dim1', 4, meta_data=ExtendedMetaData)
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()
        prob.link_phases()

        prob.setup()
        prob.set_initial_guesses()

        # Just run once to pass data.
        prob.run_model()

        # Make sure the values are correct.
        yy_actual = prob.model.get_val('traj.climb.rhs_all.test.yy')
        xx = prob.model.get_val('pre_mission.test.for_climb')
        yy_expected = 2.0 * np.sum(xx) * range(len(yy_actual))
        assert_near_equal(yy_actual, yy_expected)
        zz_actual = prob.model.get_val('traj.climb.rhs_all.test.zz')
        zz_expected = 3.0 * np.sum(xx) * range(len(zz_actual))
        assert_near_equal(zz_actual, zz_expected)

        yy_actual = prob.model.get_val('traj.climb.rhs_all.test2.test2_yy')
        xx = prob.model.get_val('pre_mission.test2.test2_for_climb')
        yy_expected = 2.0 * np.sum(xx) * range(len(yy_actual))
        assert_near_equal(yy_actual, yy_expected)
        zz_actual = prob.model.get_val('traj.climb.rhs_all.test2.test2_zz')
        zz_expected = 3.0 * np.sum(xx) * range(len(zz_actual))
        assert_near_equal(zz_actual, zz_expected)

        yy_actual = prob.model.get_val('traj.cruise.rhs_all.test.yy')
        xx = prob.model.get_val('pre_mission.test.for_cruise')
        yy_expected = 2.0 * np.sum(xx) * range(len(yy_actual))
        assert_near_equal(yy_actual, yy_expected)
        zz_actual = prob.model.get_val('traj.cruise.rhs_all.test.zz')
        zz_expected = 3.0 * np.sum(xx) * range(len(zz_actual))
        assert_near_equal(zz_actual, zz_expected)

        yy_actual = prob.model.get_val('traj.cruise.rhs_all.test2.test2_yy')
        xx = prob.model.get_val('pre_mission.test2.test2_for_cruise')
        yy_expected = 2.0 * np.sum(xx) * range(len(yy_actual))
        assert_near_equal(yy_actual, yy_expected)
        zz_actual = prob.model.get_val('traj.cruise.rhs_all.test2.test2_zz')
        zz_expected = 3.0 * np.sum(xx) * range(len(zz_actual))
        assert_near_equal(zz_actual, zz_expected)

        yy_actual = prob.model.get_val('traj.descent.rhs_all.test.yy')
        xx = prob.model.get_val('pre_mission.test.for_descent')
        yy_expected = 2.0 * np.sum(xx) * range(len(yy_actual))
        assert_near_equal(yy_actual, yy_expected)
        zz_actual = prob.model.get_val('traj.descent.rhs_all.test.zz')
        zz_expected = 3.0 * np.sum(xx) * range(len(zz_actual))
        assert_near_equal(zz_actual, zz_expected)

        # Only climb should have the zz and velocity outputs connected to post-mission.
        zzz_actual = prob.model.get_val('test.climb_zzz')
        xx = prob.model.get_val('pre_mission.test.for_climb')
        zz = prob.model.get_val('traj.climb.mission_bus_variables.zz')
        velocity = prob.model.get_val('traj.climb.mission_bus_variables.velocity')
        zzz_expected = np.sum(xx) * np.sum(zz * velocity)
        assert_near_equal(zzz_actual, zzz_expected)

        zzz_actual = prob.model.get_val('test2.climb_test2_zzz')
        xx = prob.model.get_val('pre_mission.test2.test2_for_climb')
        zz = prob.model.get_val('traj.climb.mission_bus_variables.test2_zz')
        velocity = prob.model.get_val('traj.climb.mission_bus_variables.velocity')
        zzz_expected = np.sum(xx) * np.sum(zz * velocity)
        assert_near_equal(zzz_actual, zzz_expected)

        zzz_actual = prob.model.get_val('test.cruise_zzz')
        xx = prob.model.get_val('pre_mission.test.for_cruise')
        zzz_expected = np.sum(xx)
        assert_near_equal(zzz_actual, zzz_expected)

        zzz_actual = prob.model.get_val('test2.cruise_test2_zzz')
        xx = prob.model.get_val('pre_mission.test2.test2_for_cruise')
        zzz_expected = np.sum(xx)
        assert_near_equal(zzz_actual, zzz_expected)

        zzz_actual = prob.model.get_val('test.descent_zzz')
        xx = prob.model.get_val('pre_mission.test.for_descent')
        zzz_expected = np.sum(xx)
        assert_near_equal(zzz_actual, zzz_expected)

        zzz_actual = prob.model.get_val('test2.descent_test2_zzz')
        xx = prob.model.get_val('pre_mission.test2.test2_for_descent')
        zzz_expected = np.sum(xx)
        assert_near_equal(zzz_actual, zzz_expected)


if __name__ == '__main__':
    unittest.main()
