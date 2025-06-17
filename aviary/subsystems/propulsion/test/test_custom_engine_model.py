import unittest

import dymos as dm
import openmdao.api as om
from dymos.transcriptions.transcription_base import TranscriptionBase
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic


class PreMissionEngine(om.Group):
    def setup(self):
        self.add_subsystem(
            'dummy_comp',
            om.ExecComp('y=x**2', x={'units': 'm', 'val': 2.0}, y={'units': 'm**2'}),
            promotes=['*'],
        )


class SimpleEngine(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # add inputs and outputs to interpolator
        self.add_input(
            Dynamic.Atmosphere.MACH,
            shape=nn,
            units='unitless',
            desc='Current flight Mach number',
        )
        self.add_input(
            Dynamic.Mission.ALTITUDE,
            shape=nn,
            units='ft',
            desc='Current flight altitude',
        )
        self.add_input(
            Dynamic.Vehicle.Propulsion.THROTTLE,
            shape=nn,
            units='unitless',
            desc='Current engine throttle',
        )
        self.add_input(
            'different_throttle',
            shape=nn,
            units='unitless',
            desc='Little bonus throttle for testing',
        )
        self.add_input('y', units='m**2', desc='Dummy variable for bus testing')

        self.add_output(
            Dynamic.Vehicle.Propulsion.THRUST,
            shape=nn,
            units='lbf',
            desc='Current net thrust produced (scaled)',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.THRUST_MAX,
            shape=nn,
            units='lbf',
            desc='Current net thrust produced (scaled)',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE,
            shape=nn,
            units='lbm/s',
            desc='Current fuel flow rate (scaled)',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.ELECTRIC_POWER_IN,
            shape=nn,
            units='W',
            desc='Current electric energy rate (scaled)',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.NOX_RATE,
            shape=nn,
            units='lbm/s',
            desc='Current NOx emission rate (scaled)',
        )
        self.add_output(
            Dynamic.Vehicle.Propulsion.TEMPERATURE_T4,
            shape=nn,
            units='degR',
            desc='Current turbine exit temperature',
        )

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        combined_throttle = (
            inputs[Dynamic.Vehicle.Propulsion.THROTTLE] + inputs['different_throttle']
        )

        # calculate outputs
        outputs[Dynamic.Vehicle.Propulsion.THRUST] = 10000.0 * combined_throttle
        outputs[Dynamic.Vehicle.Propulsion.THRUST_MAX] = 10000.0
        outputs[Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE] = -10.0 * combined_throttle
        outputs[Dynamic.Vehicle.Propulsion.TEMPERATURE_T4] = 2800.0


class SimpleTestEngine(EngineModel):
    def __init__(self, name='engine', options=None):
        aviary_inputs = AviaryValues()
        super().__init__(name, options=aviary_inputs)

    def build_pre_mission(self, aviary_inputs=AviaryValues()):
        return PreMissionEngine()

    def build_mission(self, num_nodes, aviary_inputs):
        return SimpleEngine(num_nodes=num_nodes)

    def get_controls(self, **kwargs):
        controls_dict = {
            'different_throttle': {
                'units': 'unitless',
                'lower': 0.0,
                'upper': 0.1,
                'control_type': 'polynomial',
                'order': 3,
            },
        }
        return controls_dict

    def get_pre_mission_bus_variables(self, aviary_inputs):
        bus_dict = {
            'y': {
                'mission_name': 'y',
                'units': 'm**2',
            },
        }
        return bus_dict

    def get_initial_guesses(self):
        initial_guesses_dict = {
            'different_throttle': {
                'val': 0.05,
                'units': 'unitless',
                'type': 'control',
            }
        }
        return initial_guesses_dict


@use_tempdirs
class CustomEngineTest(unittest.TestCase):
    def test_custom_engine(self):
        phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [],
                'optimize_mass': True,
            },
            'cruise': {
                'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
                'user_options': {
                    'num_segments': 2,
                    'order': 3,
                    'mach_optimize': False,
                    'mach_polynomial_order': 1,
                    'mach_initial': (0.72, 'unitless'),
                    'mach_final': (0.72, 'unitless'),
                    'mach_bounds': ((0.7, 0.74), 'unitless'),
                    'altitude_optimize': False,
                    'altitude_polynomial_order': 1,
                    'altitude_initial': (35000.0, 'ft'),
                    'altitude_final': (35000.0, 'ft'),
                    'altitude_bounds': ((23000.0, 38000.0), 'ft'),
                    'throttle_enforcement': 'boundary_constraint',
                    'time_initial_bounds': ((0.0, 0.0), 'min'),
                    'time_duration_bounds': ((10.0, 30.0), 'min'),
                },
                'initial_guesses': {'time': ([0, 30], 'min')},
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [],
            },
        }

        prob = AviaryProblem(reports=False)

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs(
            'models/test_aircraft/aircraft_for_bench_FwFm.csv',
            phase_info,
            engine_builders=[SimpleTestEngine()],
        )

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('SLSQP', verbosity=0)

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.final_setup()

        # check that the different throttle initial guess has been set correctly
        initial_guesses = prob.get_val('traj.cruise.controls:different_throttle')[0]
        assert_near_equal(float(initial_guesses), 0.05)

        # and run mission
        dm.run_problem(prob, run_driver=True, simulate=False, make_plots=False)

        tol = 1.0e-4

        assert_near_equal(prob.get_val('traj.cruise.rhs_all.y'), 4.0, tol)


if __name__ == '__main__':
    unittest.main()
    # test = TurbopropTest()
    # test.test_turboprop()
