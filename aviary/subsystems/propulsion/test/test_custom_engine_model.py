import unittest
import openmdao

import dymos as dm
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from packaging import version

from aviary.subsystems.propulsion.engine_model import EngineModel
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Dynamic
import dymos as dm
import unittest
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs


class PreMissionEngine(om.Group):
    def setup(self):
        self.add_subsystem('dummy_comp', om.ExecComp(
            'y=x**2', x={'units': 'm', 'val': 2.}, y={'units': 'm**2'}), promotes=['*'])


class SimpleEngine(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']
        # add inputs and outputs to interpolator
        self.add_input(Dynamic.Mission.MACH,
                       shape=nn,
                       units='unitless',
                       desc='Current flight Mach number')
        self.add_input(Dynamic.Mission.ALTITUDE,
                       shape=nn,
                       units='ft',
                       desc='Current flight altitude')
        self.add_input(Dynamic.Mission.THROTTLE,
                       shape=nn,
                       units='unitless',
                       desc='Current engine throttle')
        self.add_input('different_throttle',
                       shape=nn,
                       units='unitless',
                       desc='Little bonus throttle for testing')
        self.add_input('y',
                       units='m**2',
                       desc='Dummy variable for bus testing')

        self.add_output(Dynamic.Mission.THRUST,
                        shape=nn,
                        units='lbf',
                        desc='Current net thrust produced (scaled)')
        self.add_output(Dynamic.Mission.THRUST_MAX,
                        shape=nn,
                        units='lbf',
                        desc='Current net thrust produced (scaled)')
        self.add_output(Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE,
                        shape=nn,
                        units='lbm/s',
                        desc='Current fuel flow rate (scaled)')
        self.add_output(Dynamic.Mission.ELECTRIC_POWER,
                        shape=nn,
                        units='W',
                        desc='Current electric energy rate (scaled)')
        self.add_output(Dynamic.Mission.NOX_RATE,
                        shape=nn,
                        units='lbm/s',
                        desc='Current NOx emission rate (scaled)')
        self.add_output(Dynamic.Mission.TEMPERATURE_ENGINE_T4,
                        shape=nn,
                        units='degR',
                        desc='Current turbine exit temperature')

        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        combined_throttle = inputs[Dynamic.Mission.THROTTLE] + \
            inputs['different_throttle']

        # calculate outputs
        outputs[Dynamic.Mission.THRUST] = 10000. * combined_throttle
        outputs[Dynamic.Mission.THRUST_MAX] = 10000.
        outputs[Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE] = -10. * combined_throttle
        outputs[Dynamic.Mission.TEMPERATURE_ENGINE_T4] = 2800.


class SimpleTestEngine(EngineModel):
    def __init__(self, name='engine', options=None):
        aviary_inputs = AviaryValues()
        super().__init__(name, options=aviary_inputs)

    def build_pre_mission(self, aviary_inputs=AviaryValues()):
        return PreMissionEngine()

    def build_mission(self, num_nodes, aviary_inputs):
        return SimpleEngine(num_nodes=num_nodes)

    def get_controls(self):
        controls_dict = {
            "different_throttle": {'units': 'unitless', 'lower': 0., 'upper': 0.1},
        }
        return controls_dict

    def get_bus_variables(self):
        bus_dict = {
            "y": {
                "mission_name": "y",
                "units": "m**2",
            },
        }
        return bus_dict

    def get_initial_guesses(self):
        initial_guesses_dict = {
            "different_throttle": {
                "val": 0.05,
                "units": "unitless",
                "type": "control",
            }
        }
        return initial_guesses_dict


@use_tempdirs
class CustomEngineTest(unittest.TestCase):
    def test_custom_engine(self):

        import pkg_resources

        from aviary.interface.methods_for_level2 import AviaryProblem

        aero_data = "subsystems/aerodynamics/gasp_based/data/large_single_aisle_1_aero_free.txt"

        phase_info = {
            'pre_mission': {
                'include_takeoff': False,
                'external_subsystems': [],
                'optimize_mass': True,
            },
            'cruise': {
                'subsystem_options': {
                    'core_aerodynamics': {'method': 'solved_alpha', 'aero_data': aero_data, 'training_data': False}
                },
                'external_subsystems': [],
                'user_options': {
                    'fix_initial': False,
                    'fix_final': False,
                    'fix_duration': False,
                    'num_segments': 2,
                    'order': 3,
                    'initial_ref': (1., 's'),
                    'initial_bounds': ((0., 0.), 's'),
                    'duration_ref': (21.e3, 's'),
                    'duration_bounds': ((1.e3, 10.e3), 's'),
                    'min_altitude': (10.668e3, 'm'),
                    'max_altitude': (10.668e3, 'm'),
                    'min_mach': 0.8,
                    'max_mach': 0.8,
                    'required_available_climb_rate': (1.524, 'm/s'),
                    'input_initial': False,
                    'mass_f_cruise': (1.e4, 'lbm'),
                    'range_f_cruise': (1.e6, 'm'),
                },
                'initial_guesses': {
                    'times': ([0., 30.], 'min'),
                    'altitude': ([35.e3, 35.e3], 'ft'),
                    'velocity': ([455.49, 455.49], 'kn'),
                    'mass': ([130.e3, 128.e3], 'lbm'),
                    'range': ([0., 300.], 'nmi'),
                    'velocity_rate': ([0., 0.], 'm/s**2'),
                    'throttle': ([0.6, 0.6], 'unitless'),
                }
            },
            'post_mission': {
                'include_landing': False,
                'external_subsystems': [],
            }
        }

        csv_path = pkg_resources.resource_filename(
            "aviary", "models/test_aircraft/aircraft_for_bench_GwFm.csv")

        prob = AviaryProblem(reports=False)

        # Load aircraft and options data from user
        # Allow for user overrides here
        prob.load_inputs(csv_path, phase_info, engine_builder=SimpleTestEngine())

        # Have checks for clashing user inputs
        # Raise warnings or errors depending on how clashing the issues are
        prob.check_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()

        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver("SLSQP")

        prob.add_design_variables()

        prob.add_objective('fuel_burned')

        prob.setup()

        prob.set_initial_guesses()

        prob.final_setup()

        # check that the different throttle initial guess has been set correctly
        initial_guesses = prob.get_val(
            'traj.phases.cruise.controls:different_throttle')[0]
        assert_near_equal(float(initial_guesses), 0.05)

        # and run mission, and dynamics
        dm.run_problem(prob, run_driver=True, simulate=False, make_plots=False)

        tol = 1.e-4

        assert_near_equal(float(prob.get_val('traj.cruise.rhs_all.y')), 4., tol)

        prob_vars = prob.list_problem_vars(
            print_arrays=True, driver_scaling=False, out_stream=None)

        design_vars_dict = dict(prob_vars['design_vars'])

        # List of all expected variable names in design_vars
        expected_var_names = [
            'traj.phases.cruise.indep_states.states:altitude',
            'traj.phases.cruise.indep_states.states:velocity',
            'traj.phases.cruise.indep_states.states:mass',
            'traj.phases.cruise.indep_states.states:range',
        ]

        # Check that all expected variable names are present in design_vars
        for var_name in expected_var_names:
            self.assertIn(var_name, design_vars_dict)

        # Check values
        assert_near_equal(design_vars_dict['traj.phases.cruise.indep_states.states:altitude']['val'], [
            10668.] * 7, tolerance=tol)
        assert_near_equal(design_vars_dict['traj.phases.cruise.indep_states.states:velocity']['val'], [
            234.3243] * 7, tolerance=tol)
        # print the mass and range
        assert_near_equal(design_vars_dict['traj.phases.cruise.indep_states.states:mass']['val'], [
            58967.0081, 58805.95966377, 58583.74569223, 58513.41573, 58352.36729377, 58130.15332223, 58059.82336], tolerance=tol)
        assert_near_equal(design_vars_dict['traj.phases.cruise.indep_states.states:range']['val'], [
            0., 98633.17494548, 234726.82505452, 277800., 376433.17494548, 512526.82505452, 555600.], tolerance=tol)


if __name__ == '__main__':
    unittest.main()
