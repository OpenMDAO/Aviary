import importlib
import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.default_phase_info.two_dof_fiti import add_default_sgm_args
from aviary.interface.methods_for_level2 import AviaryGroup
from aviary.mission.flops_based.phases.time_integration_phases import SGMHeightEnergy
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
from aviary.variable_info.enums import EquationsOfMotion
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings


@use_tempdirs
@unittest.skipUnless(
    importlib.util.find_spec('pyoptsparse') is not None, 'pyoptsparse is not installed'
)
class HE_SGMDescentTestCase(unittest.TestCase):
    """
    This test builds height-energy based trajectories and then simulates them and checks that the final values are correct.
    The trajectories used are intended to be single phases to simplify debugging and to allow for easier testing of trigger based values.
    """

    def setUp(self):
        aviary_inputs, initialization_guesses = create_vehicle(
            'models/test_aircraft/aircraft_for_bench_FwFm.csv'
        )
        aviary_inputs.set_val(Aircraft.Engine.SCALED_SLS_THRUST, val=28690, units='lbf')
        aviary_inputs.set_val(Aircraft.Engine.SCALE_FACTOR, val=0.9917)
        aviary_inputs.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, val=0, units='unitless')
        aviary_inputs.set_val(
            Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT, val=0.0175, units='unitless'
        )
        aviary_inputs.set_val(
            Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT, val=0.35, units='unitless'
        )
        aviary_inputs.set_val(Settings.EQUATIONS_OF_MOTION, val=EquationsOfMotion.SOLVED_2DOF)

        engines = [build_engine_deck(aviary_inputs)]
        # don't need mass
        core_subsystems = get_default_premission_subsystems('FLOPS', engines)[:-1]
        ode_args = dict(aviary_options=aviary_inputs, core_subsystems=core_subsystems)
        preprocess_propulsion(aviary_inputs, engines)

        ode_args['num_nodes'] = 1
        ode_args['subsystem_options'] = {'core_aerodynamics': {'method': 'computed'}}

        self.ode_args = ode_args
        self.aviary_inputs = aviary_inputs
        self.tol = 1e-5

    def setup_prob(self, phases) -> om.Problem:
        prob = om.Problem()
        prob.driver = om.pyOptSparseDriver()
        prob.driver.options['optimizer'] = 'IPOPT'
        prob.driver.opt_settings['tol'] = 1.0e-6
        prob.driver.opt_settings['mu_init'] = 1e-5
        prob.driver.opt_settings['max_iter'] = 50
        prob.driver.opt_settings['print_level'] = 5

        aviary_options = self.ode_args['aviary_options']
        subsystems = self.ode_args['core_subsystems']

        add_default_sgm_args(phases, self.ode_args)

        traj = FlexibleTraj(
            Phases=phases,
            promote_all_auto_ivc=True,
            traj_final_state_output=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
            traj_initial_state_input=[
                Dynamic.Vehicle.MASS,
                Dynamic.Mission.DISTANCE,
                Dynamic.Mission.ALTITUDE,
            ],
        )
        prob.model = AviaryGroup(aviary_options=aviary_options, aviary_metadata=BaseMetaData)
        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_options, subsystems=subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )
        prob.model.add_subsystem('traj', traj, promotes=['aircraft:*', 'mission:*'])

        prob.model.add_subsystem(
            'fuel_obj',
            om.ExecComp(
                'reg_objective = overall_fuel/10000',
                reg_objective={'val': 0.0, 'units': 'unitless'},
                overall_fuel={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('overall_fuel', Mission.Summary.TOTAL_FUEL_MASS),
            ],
            promotes_outputs=[('reg_objective', Mission.Objectives.FUEL)],
        )

        prob.model.add_objective(Mission.Objectives.FUEL, ref=1e4)

        setup_model_options(prob, aviary_options)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)

            prob.setup()

        return prob

    def run_simulation(self, phases, initial_values: dict):
        prob = self.setup_prob(phases)

        for key, val in initial_values.items():
            prob.set_val(key, **val)

        prob.run_model()

        distance = prob.get_val('traj.distance_final', units='NM')[0]
        mass = prob.get_val('traj.mass_final', units='lbm')[0]
        alt = prob.get_val('traj.altitude_final', units='ft')[0]

        final_states = {'distance': distance, 'mass': mass, 'altitude': alt}
        return final_states

    # def test_takeoff(self):
    #     initial_values_takeoff = {
    #         "traj.altitude_initial": {'val': 0, 'units': 'ft'},
    #         "traj.mass_initial": {'val': 171000, 'units': "lbm"},
    #         "traj.distance_initial": {'val': 0, 'units': "NM"},
    #         "traj.velocity": {'val': .1, 'units': "m/s"},
    #     }

    #     ode_args = self.ode_args
    #     ode_args['friction_key'] = Mission.Takeoff.ROLLING_FRICTION_COEFFICIENT
    #     brake_release_to_decision = SGMDetailedTakeoff(
    #         ode_args,
    #         simupy_args=dict(verbosity=Verbosity.DEBUG,)
    #         )
    #     brake_release_to_decision.clear_triggers()
    #     brake_release_to_decision.add_trigger(Dynamic.Mission.VELOCITY, value=167.85, units='kn')

    #     phases = {'HE': {
    #         'ode': brake_release_to_decision,
    #         'vals_to_set': {}
    #     }}

    #     final_states = self.run_simulation(phases, initial_values_takeoff)
    #     # assert_near_equal(final_states['altitude'], 500, self.tol)
    #     assert_near_equal(final_states['velocity'], 167.85, self.tol)

    def test_cruise(self):
        initial_values_cruise = {
            'traj.altitude_initial': {'val': 35000, 'units': 'ft'},
            'traj.mass_initial': {'val': 171000, 'units': 'lbm'},
            'traj.distance_initial': {'val': 0, 'units': 'NM'},
            'traj.mach': {'val': 0.8, 'units': 'unitless'},
        }

        phases = {
            'HE': {
                'kwargs': {
                    'mass_trigger': (160000, 'lbm'),
                },
                'builder': SGMHeightEnergy,
                'user_options': {},
            }
        }

        final_states = self.run_simulation(phases, initial_values_cruise)
        assert_near_equal(final_states['mass'], 160000, self.tol)

    # def test_landing(self):
    #     initial_values_landing = {
    #         "traj.altitude_initial": {'val': 35000, 'units': 'ft'},
    #         "traj.mass_initial": {'val': 171000, 'units': "lbm"},
    #         "traj.distance_initial": {'val': 0, 'units': "NM"},
    #         "traj.velocity": {'val': 300, 'units': "m/s"},
    #     }

    #     ode_args = self.ode_args
    #     ode_args['friction_key'] = Mission.Takeoff.BRAKING_FRICTION_COEFFICIENT
    #     phases = {'HE': {
    #         'ode': SGMDetailedLanding(
    #             ode_args,
    #             simupy_args=dict(verbosity=Verbosity.QUIET,)
    #         ),
    #         'vals_to_set': {}
    #     }}

    #     final_states = self.run_simulation(phases, initial_values_landing)
    #     assert_near_equal(final_states['altitude'], 0, self.tol)


if __name__ == '__main__':
    unittest.main()
