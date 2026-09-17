import unittest
from copy import deepcopy

import openmdao.api as om

import aviary.api as av
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.subsystems.test.subsystem_tester import TestSubsystemBuilder
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


class FullVariableSet:
    class Dummy:
        DUMMY_CONSTRAINT_VARIABLE = 'aircraft:dummy_constraint_variable'
        DUMMY_CONTROL_VARIABLE = 'aircraft:dummy_control_variable'
        DUMMY_DESIGN_VARIABLE = 'aircraft:dummy_design_variable'
        DUMMY_STATE_VARIABLE = 'aircraft:dummy_state_variable'
        DUMMY_PARAMETER_VARIABLE = 'aircraft:dummy_parameter_variable'
        DUMMY_PRE_MISSION_INPUT = 'aircraft:dummy_pre_mission_input'
        DUMMY_PRE_MISSION_OUTPUT = 'aircraft:dummy_pre_mission_output'
        DUMMY_MISSION_INPUT = 'aircraft:dummy_mission_input'
        DUMMY_MISSION_OUTPUT = 'aircraft:dummy_mission_output'
        DUMMY_POST_MISSION_INPUT = 'aircraft:dummy_post_mission_input'
        DUMMY_POST_MISSION_OUTPUT = 'aircraft:dummy_post_mission_output'
        DUMMY_TIMESERIES_VARIABLE = 'aircraft:dummy_timeseries_variable'


DUMMY_PRE_MISSION_BUS = 'dummy_pre_mission_bus'
DUMMY_POST_MISSION_BUS = 'dummy_post_mission_bus'

ExtendedMetaData = deepcopy(av.CoreMetaData)
AdditionalMetaData = deepcopy(av.CoreMetaData)

# Variables for ExtendedMetaData
av.add_meta_data(
    Aircraft.Dummy.VARIABLE,
    desc='Dummy aircraft variable',
    default_value=0.25,
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

av.add_meta_data(
    FullVariableSet.Dummy.DUMMY_DESIGN_VARIABLE,
    desc='Dummy design variable',
    default_value=0.5,
    option=False,
    units='A',
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

        # This explicit component is returned in build_pre_mission, which simply calculates an output from an input


class DummyFullPreMissionComp(om.ExplicitComponent):
    def setup(self):
        self.add_input(FullVariableSet.Dummy.DUMMY_DESIGN_VARIABLE, units='A')
        self.add_input(FullVariableSet.Dummy.DUMMY_PRE_MISSION_INPUT, units='kn')
        self.add_output(FullVariableSet.Dummy.DUMMY_PRE_MISSION_OUTPUT, units='kg')
        self.add_output(FullVariableSet.Dummy.DUMMY_MISSION_OUTPUT, units='kg')
        self.add_output(DUMMY_PRE_MISSION_BUS, units='kg')
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[FullVariableSet.Dummy.DUMMY_PRE_MISSION_OUTPUT] = (
            2
            * inputs[FullVariableSet.Dummy.DUMMY_PRE_MISSION_INPUT]
            * inputs[FullVariableSet.Dummy.DUMMY_DESIGN_VARIABLE]
        )
        outputs[FullVariableSet.Dummy.DUMMY_MISSION_OUTPUT] = 1
        outputs[DUMMY_PRE_MISSION_BUS] = 1


# This Explicit Component is returned in the build_mission function when defining the SubsystemBuilder. From Documentation
# need to provide computations for state rates so the mission integration code can compute state values.
class DummyFullMissionComp(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input(FullVariableSet.Dummy.DUMMY_MISSION_INPUT, units='m**2', shape=nn)
        self.add_input(FullVariableSet.Dummy.DUMMY_STATE_VARIABLE, units='m/s', shape=nn)
        self.add_input(FullVariableSet.Dummy.DUMMY_PARAMETER_VARIABLE, units='m', shape=nn)
        self.add_input(FullVariableSet.Dummy.DUMMY_CONTROL_VARIABLE, units='unitless', shape=nn)
        self.add_input(DUMMY_PRE_MISSION_BUS, units='kg', shape=1)
        self.add_output(FullVariableSet.Dummy.DUMMY_CONSTRAINT_VARIABLE, units='unitless', shape=nn)
        self.add_output(FullVariableSet.Dummy.DUMMY_MISSION_OUTPUT, units='kg', shape=nn)
        self.add_output(
            FullVariableSet.Dummy.DUMMY_STATE_VARIABLE + '_rate', units='m/s**2', shape=nn
        )
        self.add_output(FullVariableSet.Dummy.DUMMY_TIMESERIES_VARIABLE, units='unitless', shape=nn)
        self.add_output(DUMMY_POST_MISSION_BUS, units='kg', shape=nn)

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[FullVariableSet.Dummy.DUMMY_MISSION_OUTPUT] = (
            2 * inputs[FullVariableSet.Dummy.DUMMY_MISSION_INPUT]
        )
        outputs[FullVariableSet.Dummy.DUMMY_STATE_VARIABLE + '_rate'] = 1
        outputs[FullVariableSet.Dummy.DUMMY_CONSTRAINT_VARIABLE] = inputs[
            FullVariableSet.Dummy.DUMMY_CONTROL_VARIABLE
        ]
        outputs[FullVariableSet.Dummy.DUMMY_TIMESERIES_VARIABLE] = 1
        outputs[DUMMY_POST_MISSION_BUS][:] = inputs[DUMMY_PRE_MISSION_BUS]


# Almost identical ot the pre-mission component
class DummyFullPostMissionComp(om.ExplicitComponent):
    def setup(self):
        self.add_input(FullVariableSet.Dummy.DUMMY_POST_MISSION_INPUT, units='kn')
        self.add_input(DUMMY_POST_MISSION_BUS, units='kg', shape_by_conn=True)
        self.add_output(FullVariableSet.Dummy.DUMMY_POST_MISSION_OUTPUT, units='kg')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        outputs[FullVariableSet.Dummy.DUMMY_POST_MISSION_OUTPUT] = (
            2 * inputs[FullVariableSet.Dummy.DUMMY_POST_MISSION_INPUT]
            + inputs[DUMMY_POST_MISSION_BUS][-1]
        )


class PreOnlyBuilder(SubsystemBuilder):
    def build_pre_mission(self, aviary_inputs, subsystem_options):
        return DummyComp()

    def get_mass_names(self, aviary_inputs=None):
        return [Aircraft.Dummy.VARIABLE_OUT]


class PostOnlyBuilder(SubsystemBuilder):
    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        group = om.Group()
        group.add_subsystem(
            'comp',
            om.ExecComp('y_postmission = x**2'),
            promotes_inputs=[('x', Aircraft.Dummy.VARIABLE_OUT)],
            promotes_outputs=['*'],
        )
        return group


class FailingSubsystemBuilder(SubsystemBuilder):
    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            'State1': {
                'rate_source': 'NonExistentRateSource',
            }
        }

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return om.ExecComp('y = x**2')


class ArrayGuessSubsystemBuilder(SubsystemBuilder):
    def __init__(self, name='array_guess'):
        super().__init__(name, meta_data=ExtendedMetaData)

    def build_pre_mission(self, aviary_inputs, subsystem_options):
        return DummyComp()

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return DummyMissionComp(num_nodes=num_nodes)

    def get_initial_guesses(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            Mission.Dummy.VARIABLE: {
                'val': [1.0, 2.0, 3.0],
                'type': 'state',
                'units': 'm',
            }
        }

    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            Mission.Dummy.VARIABLE: {
                'rate_source': Mission.Dummy.VARIABLE_RATE,
                'units': 'm',
            }
        }

    def get_controls(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {}

    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None):
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


class AdditionalArrayGuessSubsystemBuilder(SubsystemBuilder):
    def __init__(self, name='additional_array_guess'):
        super().__init__(name, meta_data=AdditionalMetaData)

    def build_pre_mission(self, aviary_inputs, subsystem_options):
        return DummyWingspanComp()

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return DummyFlightDurationComp(num_nodes=num_nodes)

    # def mission_outputs(self, **kwargs):
    #     return [MoreMission.Dummy.DUMMY_FLIGHT_DURATION + '_rate']

    def get_initial_guesses(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            MoreMission.Dummy.DUMMY_FLIGHT_DURATION: {
                'val': [1.0, 2.0, 3.0],
                'type': 'state',
                'units': 'h',
            }
        }

    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            MoreMission.Dummy.DUMMY_FLIGHT_DURATION: {
                'rate_source': MoreMission.Dummy.DUMMY_FLIGHT_DURATION + '_rate',
                'units': 'h',
            }
        }

    def get_controls(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        if subsystem_options and subsystem_options.get('enable_control', False):
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

    def get_timeseries(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return [MoreMission.Dummy.DUMMY_CONTROL, MoreMission.Dummy.TIMESERIES_VAR]


class FullSubsystemBuilder(SubsystemBuilder):
    # Sets up self.name and self.meta_data.
    def __init__(self, name='full_suite'):
        super().__init__(name)

    # Copied from separate existing subsystem builder. This function returns a dictionary of dynamic
    # states. Required for subsystems with mission-based dynamics. User models must provide time derivative
    # of each state.

    # states are summed over the course of a phase by Dymos

    # Return dictionary should have rate source & units specified.

    # In this particular example we're saying DUMMY_STATE_VARIABLE is our dictionary key and
    # the value is a dictionary with the rate_source and units (of state variable) attached
    def get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            FullVariableSet.Dummy.DUMMY_STATE_VARIABLE: {
                'rate_source': FullVariableSet.Dummy.DUMMY_STATE_VARIABLE + '_rate',
                'units': 'm/s',
            }
        }

    # true if the mission subsystem needs to be in the solver loop in mission
    def needs_mission_solver(self, aviary_inputs, user_options, subsystem_options):
        return True

    # builds an OpenMDAO system for pre-mission computations.
    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        return DummyFullPreMissionComp()

    # return dictionary of control variables for subsystem

    # From 'Mission Definition' documentation, which talks about information needed for creation of a
    # Dymos trajectory (or dymos phases) - Controls are variables that are directly controlled by the optimizer

    # return is like get_states. 1st term is the variable name, dictionary second term with units,
    # opt flag (when true the control becomes an optimizer design variable), additional keywords for
    # the control variable
    def get_controls(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        controls_dict = {
            FullVariableSet.Dummy.DUMMY_CONTROL_VARIABLE: {
                'units': 'unitless',
                'opt': True,
                'lower': 0,
                'upper': 1,
            }
        }
        return controls_dict

    # Parameters are variables that are allowed to be controlled by the optimizer and are assumed to be constant
    # across the entire trajectory or a single phase
    def get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {FullVariableSet.Dummy.DUMMY_PARAMETER_VARIABLE: {'val': 2.0, 'units': 'm'}}

    # place limits on a part of a phase? What is allowed to be constrained? Which variables?
    def get_constraints(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        constraints = {
            FullVariableSet.Dummy.DUMMY_CONSTRAINT_VARIABLE: {
                'lower': 1.0,
                'type': 'boundary',
                'loc': 'final',
            },
        }
        return constraints

    # Return a list of variable names that will be linked between phases...
    def get_linked_variables(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return [FullVariableSet.Dummy.DUMMY_STATE_VARIABLE]

    # Might not need this
    # def get_bus_variables(self, aviary_inputs=None):
    #    return super().get_bus_variables(aviary_inputs)

    # returns a dictionary of variables that are passed from pre-mission to mission to post-mission
    # Are these linked? Here I've specified 'y' as pre-mission with the same name in mission?
    def get_pre_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        bus_dict = {
            'full_suite.' + DUMMY_PRE_MISSION_BUS: {
                'mission_name': 'full_suite.' + DUMMY_PRE_MISSION_BUS,
                'units': 'kg',
            },
        }
        return bus_dict

    # Build OpenMDAO System for mission computations of the subsystem
    # Same scheme as build pre or post mission. For now just tossing in a random computation
    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        return DummyFullMissionComp(num_nodes=num_nodes)

    # List of inputs to be promoted out of the external subsystem
    # Isn't promotion just linking variables with the same name? How does this differ from linked variables or
    # get pre/post mission variables?
    def mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        inputs = [
            FullVariableSet.Dummy.DUMMY_MISSION_INPUT,
            FullVariableSet.Dummy.DUMMY_PARAMETER_VARIABLE,
            FullVariableSet.Dummy.DUMMY_CONTROL_VARIABLE,
        ]
        return inputs

    # same as above but with outputs... Should this list match get_post_mission_bus_variables
    def mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        outputs = [
            FullVariableSet.Dummy.DUMMY_MISSION_OUTPUT,
            FullVariableSet.Dummy.DUMMY_STATE_VARIABLE + '_rate',
            FullVariableSet.Dummy.DUMMY_TIMESERIES_VARIABLE,
            FullVariableSet.Dummy.DUMMY_CONSTRAINT_VARIABLE,
            DUMMY_POST_MISSION_BUS,
        ]
        return outputs

    # return design variables... which are variables modified by optimizer? that right?
    def get_design_vars(self, aviary_inputs=None):
        DVs = {
            FullVariableSet.Dummy.DUMMY_DESIGN_VARIABLE: {
                'units': 'A',
                'lower': 0.0,
                'upper': 2.0,
            },
        }
        return DVs

    # get initial guesses, pretty straight forward. Which variables are set with initial guesses?
    def get_initial_guesses(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return {
            FullVariableSet.Dummy.DUMMY_STATE_VARIABLE: {
                'val': 10.0,
                'type': 'state',
                'units': 'm/s',
            }
        }

    # Return a list of names of mass variables
    def get_mass_names(self, aviary_inputs=None):
        return [FullVariableSet.Dummy.DUMMY_MISSION_OUTPUT]

    # Preprocess inputs to the subsystem, returning a modified AviaryValues object
    # modifies the inputs before they are set in the subsystem...
    def preprocess_inputs(self, aviary_inputs=None):
        aviary_inputs.set_val(FullVariableSet.Dummy.DUMMY_MISSION_INPUT, 5.0, 'm**2')
        return aviary_inputs

    # Replaced with get_timeseries
    # def get_outputs(self):
    #    return super().get_outputs()

    # Returns a list of outputs to add to the Dymos timeseries outputs for use when graphing or
    # post-processing the mission.
    def get_timeseries(self, aviary_inputs=None, user_options=None, subsystem_options=None):
        return [FullVariableSet.Dummy.DUMMY_TIMESERIES_VARIABLE]

    # Return a dict mapping phase names to a dict mapping mission variable names to post_mission variable names
    def get_post_mission_bus_variables(self, aviary_inputs=None, mission_info=None):
        bus_dict = {
            'climb': {
                DUMMY_POST_MISSION_BUS: {
                    'post_mission_name': 'full_suite.' + DUMMY_POST_MISSION_BUS
                },
            }
        }
        return bus_dict

    #
    def build_post_mission(
        self,
        aviary_inputs=None,
        mission_info=None,
        subsystem_options=None,
        phase_mission_bus_lengths=None,
    ):
        return DummyFullPostMissionComp()

    def report(self, prob, reports_folder, **kwargs):
        filename = 'FullSubsystemTest.md'
        filepath = filename

        with open(filepath, mode='w') as f:
            f.write(f'Test Report Written')


class TestPreOnly(TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = PreOnlyBuilder()


class TestPostOnly(TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = PostOnlyBuilder()


class TestFailingBuilder(TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = FailingSubsystemBuilder()

    def test_check_state_variables(self):
        with self.assertRaises(AssertionError):
            super().test_check_state_variables()


class TestArrayGuessBuilder(TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = ArrayGuessSubsystemBuilder()


class TestAdditionalArrayGuessBuilder(TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = AdditionalArrayGuessSubsystemBuilder()


class TestFullBuilder(TestSubsystemBuilder):
    def setUp(self):
        self.subsystem_builder = FullSubsystemBuilder()


if __name__ == '__main__':
    unittest.main()
    # TestFullBuilder.setUp()
