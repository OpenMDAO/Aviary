import unittest
from copy import deepcopy

import openmdao.api as om

import aviary.api as av
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilderBase
from aviary.variable_info.variables import Aircraft as av_Aircraft
from aviary.variable_info.variables import Mission as av_Mission


class Aircraft(av_Aircraft):
    class Dummy:
        VARIABLE = 'aircraft:dummy:VARIABLE'
        VARIABLE_OUT = 'aircraft:dummy:VARIABLE_OUT'
        PARAMETER = 'aircraft:dummy:PARAMETER'


class Mission(av_Mission):
    class Dummy:
        VARIABLE = 'mission:dummy:VARIABLE'
        VARIABLE_RATE = 'mission:dummy:VARIABLE_RATE'


class MoreAircraft(av_Aircraft):
    class Dummy:
        DUMMY_WINGSPAN = 'aircraft:dummy:DUMMY_WINGSPAN'
        DUMMY_AIRSPEED = 'aircraft:dummy:DUMMY_AIRSPEED'
        DUMMY_FUEL_CAPACITY = 'aircraft:dummy:DUMMY_FUEL_CAPACITY'


class MoreMission(av_Mission):
    class Dummy:
        DUMMY_FLIGHT_DURATION = 'mission:dummy:DUMMY_FLIGHT_DURATION'
        DUMMY_TAKEOFF_WEIGHT = 'mission:dummy:DUMMY_TAKEOFF_WEIGHT'
        DUMMY_CONTROL = 'mission:dummy:DUMMY_CONTROL'
        TIMESERIES_VAR = 'mission:dummy:TIMESERIES_VAR'


ExtendedMetaData = deepcopy(av.CoreMetaData)
AdditionalMetaData = deepcopy(av.CoreMetaData)

# Variables for ExtendedMetaData
av.add_meta_data(
    Aircraft.Dummy.VARIABLE,
    desc='Dummy aircraft variable',
    default_value=0.5,
    option=False,
    units='kn',
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Dummy.VARIABLE_OUT,
    desc='Dummy aircraft variable out',
    default_value=0.5,
    option=False,
    units='kg',
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Aircraft.Dummy.PARAMETER,
    desc='Dummy mission parameter',
    default_value=0.5,
    option=False,
    units='m',
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Mission.Dummy.VARIABLE,
    desc='Dummy mission variable',
    default_value=0.5,
    option=False,
    units='m',
    meta_data=ExtendedMetaData,
)

av.add_meta_data(
    Mission.Dummy.VARIABLE_RATE,
    desc='Dummy mission variable rate',
    default_value=0.5,
    option=False,
    units='m/s',
    meta_data=ExtendedMetaData,
)

# Variables for AdditionalMetaData
av.add_meta_data(
    MoreAircraft.Dummy.DUMMY_WINGSPAN,
    desc='Dummy wingspan variable',
    default_value=35.0,
    option=False,
    units='m',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreAircraft.Dummy.DUMMY_WINGSPAN + '_out',
    desc='Output of dummy wingspan variable',
    default_value=70.0,
    option=False,
    units='m',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreAircraft.Dummy.DUMMY_AIRSPEED,
    desc='Dummy airspeed variable',
    default_value=500.0,
    option=False,
    units='kn',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreAircraft.Dummy.DUMMY_AIRSPEED + '_out',
    desc='Output of dummy airspeed variable',
    default_value=1000.0,
    option=False,
    units='kn',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreAircraft.Dummy.DUMMY_FUEL_CAPACITY,
    desc='Dummy fuel capacity variable',
    default_value=20000.0,
    option=False,
    units='kg',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreAircraft.Dummy.DUMMY_FUEL_CAPACITY + '_out',
    desc='Output of dummy fuel capacity variable',
    default_value=40000.0,
    option=False,
    units='kg',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreMission.Dummy.DUMMY_FLIGHT_DURATION,
    desc='Dummy flight duration variable',
    default_value=6.0,
    option=False,
    units='hr',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreMission.Dummy.DUMMY_FLIGHT_DURATION + '_rate',
    desc='Rate of dummy flight duration variable',
    default_value=12.0,
    option=False,
    units='hr/s',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreMission.Dummy.DUMMY_TAKEOFF_WEIGHT,
    desc='Dummy takeoff weight variable',
    default_value=80000.0,
    option=False,
    units='kg',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreMission.Dummy.DUMMY_TAKEOFF_WEIGHT + '_rate',
    desc='Rate of dummy takeoff weight variable',
    default_value=160000.0,
    option=False,
    units='kg/s',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreMission.Dummy.DUMMY_CONTROL,
    desc='Dummy control variable',
    default_value=0.5,
    option=False,
    units='unitless',
    meta_data=AdditionalMetaData,
)

av.add_meta_data(
    MoreMission.Dummy.TIMESERIES_VAR,
    desc='Dummy timeseries variable',
    default_value=0.5,
    option=False,
    units='unitless',
    meta_data=AdditionalMetaData,
)


class DummyComp(om.ExplicitComponent):
    def setup(self):
        self.add_input(Aircraft.Dummy.VARIABLE, units='kn')
        self.add_output(Aircraft.Dummy.VARIABLE_OUT, units='kg')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[Aircraft.Dummy.VARIABLE_OUT] = 2 * inputs[Aircraft.Dummy.VARIABLE]


class DummyMissionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(Mission.Dummy.VARIABLE, units='m', shape=nn)
        self.add_input(Aircraft.Dummy.PARAMETER, units='m')
        self.add_output(Mission.Dummy.VARIABLE_RATE, units='m/s', shape=nn)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[Mission.Dummy.VARIABLE_RATE] = (
            2 * inputs[Mission.Dummy.VARIABLE] + inputs[Aircraft.Dummy.PARAMETER] * 0.5
        )


class PreOnlyBuilder(SubsystemBuilderBase):
    def build_pre_mission(self, aviary_inputs):
        group = om.Group()
        group.add_subsystem('comp', DummyComp(), promotes=['*'])
        return group

    def get_mass_names(self):
        return [Aircraft.Dummy.VARIABLE_OUT]


class PostOnlyBuilder(SubsystemBuilderBase):
    def build_post_mission(self, aviary_inputs, phase_info, phase_mission_bus_lengths):
        group = om.Group()
        group.add_subsystem('comp', om.ExecComp('y = x**2'), promotes=['*'])
        return group


class FailingSubsystemBuilder(SubsystemBuilderBase):
    def get_states(self):
        return {
            'State1': {
                'rate_source': 'NonExistentRateSource',
            }
        }

    def build_mission(self, num_nodes, aviary_inputs):
        group = om.Group()
        group.add_subsystem('comp', om.ExecComp('y = x**2'))
        return group


class ArrayGuessSubsystemBuilder(SubsystemBuilderBase):
    def __init__(self, name='array_guess'):
        super().__init__(name, meta_data=ExtendedMetaData)

    def build_pre_mission(self, aviary_inputs):
        group = om.Group()
        group.add_subsystem('comp', DummyComp(), promotes=['*'])
        return group

    def build_mission(self, num_nodes, aviary_inputs):
        group = om.Group()
        group.add_subsystem('comp', DummyMissionComp(num_nodes=num_nodes), promotes=['*'])
        return group

    def get_initial_guesses(self):
        return {
            Mission.Dummy.VARIABLE: {
                'val': [1.0, 2.0, 3.0],
                'type': 'state',
                'units': 'm',
            }
        }

    def get_states(self):
        return {
            Mission.Dummy.VARIABLE: {
                'rate_source': Mission.Dummy.VARIABLE_RATE,
                'units': 'm',
            }
        }

    def get_controls(self, **kwargs):
        return {}

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        return {Aircraft.Dummy.PARAMETER: {'val': 2.0, 'units': 'm'}}


class DummyWingspanComp(om.ExplicitComponent):
    def setup(self):
        self.add_input(MoreAircraft.Dummy.DUMMY_WINGSPAN, units='m')
        self.add_output(MoreAircraft.Dummy.DUMMY_WINGSPAN + '_out', units='m')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[MoreAircraft.Dummy.DUMMY_WINGSPAN + '_out'] = (
            2 * inputs[MoreAircraft.Dummy.DUMMY_WINGSPAN]
        )


class DummyFlightDurationComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(MoreMission.Dummy.DUMMY_FLIGHT_DURATION, units='h', shape=nn)
        self.add_input(MoreMission.Dummy.DUMMY_CONTROL, units='unitless', shape=nn)
        self.add_output(MoreMission.Dummy.DUMMY_FLIGHT_DURATION + '_rate', units='h/s', shape=nn)
        self.add_output(MoreMission.Dummy.TIMESERIES_VAR, units='unitless', shape=nn)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[MoreMission.Dummy.DUMMY_FLIGHT_DURATION + '_rate'] = (
            2
            * inputs[MoreMission.Dummy.DUMMY_FLIGHT_DURATION]
            * inputs[MoreMission.Dummy.DUMMY_CONTROL]
        )
        outputs[MoreMission.Dummy.TIMESERIES_VAR] = (
            inputs[MoreMission.Dummy.DUMMY_CONTROL] ** 2 + 0.5
        )


class AdditionalArrayGuessSubsystemBuilder(SubsystemBuilderBase):
    def __init__(self, name='additional_array_guess'):
        super().__init__(name, meta_data=AdditionalMetaData)

    def build_pre_mission(self, aviary_inputs):
        group = om.Group()
        group.add_subsystem('comp', DummyWingspanComp(), promotes=['*'])
        return group

    def build_mission(self, num_nodes, aviary_inputs):
        group = om.Group()
        group.add_subsystem('comp', DummyFlightDurationComp(num_nodes=num_nodes), promotes=['*'])
        return group

    def get_initial_guesses(self):
        return {
            MoreMission.Dummy.DUMMY_FLIGHT_DURATION: {
                'val': [1.0, 2.0, 3.0],
                'type': 'state',
                'units': 'h',
            }
        }

    def get_states(self):
        return {
            MoreMission.Dummy.DUMMY_FLIGHT_DURATION: {
                'rate_source': MoreMission.Dummy.DUMMY_FLIGHT_DURATION + '_rate',
                'units': 'h',
            }
        }

    def get_controls(self, phase_name=None):
        if phase_name == 'cruise':
            controls_dict = {
                MoreMission.Dummy.DUMMY_CONTROL: {
                    'units': 'unitless',
                    'opt': True,
                    'lower': 0,
                    'upper': 1,
                }
            }
        else:
            controls_dict = {}
        return controls_dict

    def get_outputs(self):
        return [MoreMission.Dummy.DUMMY_CONTROL, MoreMission.Dummy.TIMESERIES_VAR]


class TestPreOnly(TestSubsystemBuilderBase):
    def setUp(self):
        self.subsystem_builder = PreOnlyBuilder()


class TestPostOnly(TestSubsystemBuilderBase):
    def setUp(self):
        self.subsystem_builder = PostOnlyBuilder()


class TestFailingBuilder(TestSubsystemBuilderBase):
    def setUp(self):
        self.subsystem_builder = FailingSubsystemBuilder()

    def test_check_state_variables(self):
        with self.assertRaises(AssertionError):
            super().test_check_state_variables()


class TestArrayGuessBuilder(TestSubsystemBuilderBase):
    def setUp(self):
        self.subsystem_builder = ArrayGuessSubsystemBuilder()


class TestAdditionalArrayGuessBuilder(TestSubsystemBuilderBase):
    def setUp(self):
        self.subsystem_builder = AdditionalArrayGuessSubsystemBuilder()


if __name__ == '__main__':
    unittest.main()
