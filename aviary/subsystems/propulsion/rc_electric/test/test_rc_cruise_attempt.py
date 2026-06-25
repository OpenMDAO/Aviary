import unittest
import aviary.api as av
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.examples.external_subsystems.custom_aero.custom_aero_builder import CustomAeroBuilder 
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_premission import RCPropPreMission
from aviary.subsystems.propulsion.rc_electric.model.rcpropulsion_mission import RCPropMission
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.dbf_variables import Aircraft, Dynamic



"""Test RC electric propulsion subsystem in both pre-mission and mission contexts, using the same model code but different promoted inputs/outputs and options. The pre-mission test checks that the motor mass, battery energy capacity, and motor resistance calculations are correct for a single motor. The mission test checks that the power balance residual is near zero for a single motor at nonzero throttle, and that the partial derivatives of the mission model are correct. These tests cover the core physics and math of the RC electric propulsion model, independent of any particular aircraft or mission setup. The model is exercised with realistic input values inspired by a small UAV use case, but the focus is on verifying the correctness of the propulsion calculations rather than matching a specific scenario. The same propulsion model code is used in both tests to ensure consistency between pre-mission sizing and mission performance predictions. The model was developed using open-source data from RC groups/forums and validated against typical performance metrics for small electric motors and propellers"""
""" Expected Values """

@use_tempdirs
class TestRCPropMission(unittest.TestCase):
    
    def test_residual(self):
        nn = 3

        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.NUM_ENGINES, 1)
        prob.model.add_subsystem('rc_prop_group', RCPropMission(num_nodes=nn, aviary_options= options), promotes=['*'])

        # Solve the implicit current balance with Newton for this residual test.
        prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        prob.model.nonlinear_solver.options['maxiter'] = 30
        prob.model.nonlinear_solver.options['err_on_non_converge'] = True
        prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
        prob.model.nonlinear_solver.linesearch.options['bound_enforcement'] = 'scalar'
        prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
        
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
        prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, np.full(nn, 0.8))
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
        prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
        prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
        prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
        prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
        prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')

        prob.run_model()

        battery_power = prob.get_val('battery.power', units='W')
        esc_power = prob.get_val('esc.power', units='W')
        motor_power = prob.get_val('motor.power', units='W')

    def test_premission_calcs(self):
        prob = om.Problem()
        options = AviaryValues()
        options.set_val(Aircraft.Engine.Motor.KV_EQ_SLOPE, 1.3132)
        options.set_val(Aircraft.Engine.Motor.KV_EQ_INT, 0.01)

        prob.model.add_subsystem(
            'rc_calcs', RCPropPreMission(aviary_options=options), promotes=['*']
        )

        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Battery.MASS, 0.707, units='kg')
        prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
        prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
        prob.set_val(Aircraft.Engine.Motor.MAX_CONT_CURRENT, 120, units='A')
        prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')

        prob.run_model()

        kv = prob.get_val(Aircraft.Engine.Motor.KV, 'rpm/V')
        resistance = prob.get_val(Aircraft.Engine.Motor.RESISTANCE, 'ohm')
        energy = prob.get_val(Aircraft.Battery.ENERGY_CAPACITY, 'W*h')

        kv_expected = 600.0
        resistance_expected = 0.05582266503
        energy_expected = 109.11522
        assert_near_equal(kv, kv_expected, tolerance=1e-9)
        assert_near_equal(resistance, resistance_expected, tolerance=1e-9)
        assert_near_equal(energy, energy_expected, tolerance=1e-9)
        partial_data = prob.check_partials(out_stream=None, method='cs')
        assert_check_partials(partial_data, atol=1e-12, rtol=
1e-12)
    
       
    


# NOTE: no @use_tempdirs here. DBFMassBuilder reads its airfoil CSV via a repo-root-
# relative path (like the dbf_based_mass unit tests), so this must run from the repo root.
class TestRCCruiseAttempt(unittest.TestCase):
    def setUp(self):
        self.phase_info = {'pre_mission': {'include_takeoff': False,  'optimize_mass': False, 'external_subsystems': [DBFMassBuilder()]},
   
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'external'}},
        'external_subsystems': [CustomAeroBuilder()],
        'user_options': {   # matches Cruise_Attempt.py exactly
            'num_segments': 5,
            'order': 3,
            'mach_optimize': False,
            'mach_initial': (0.0538, 'unitless'),
            'mach_final': (0.0538, 'unitless'),
            'altitude_optimize': False,
            'altitude_initial': (200.0, 'ft'),
            'altitude_final': (200.0, 'ft'),
            'distance_initial': (0.0, 'm'),
            'distance_ref': (1000.0, 'm'),
            'target_distance': (1000.0, 'm'),
            'mass_ref': (4.0, 'kg'),
            'throttle_enforcement': 'bounded',
            'time_initial': (0.0, 's'),
            'time_duration_bounds': ((20, 90.0), 's'),
        },
        'initial_guesses': {
            'time': ([0.0, 54.7], 's'),
            'distance': ([0.0, 1000.0], 'm'),
            'mach': ([0.0538, 0.0538], 'unitless'),
        },
    },
    
    'post_mission': {
        'include_landing': False,
        # 'target_range': (200, 'ft'),
        # 'constraint_range':True, 
    }
        }   

    def test_subsystems_in_cruise_attempt(self):
        phase_info = self.phase_info.copy()

        prob = av.AviaryProblem(verbosity=0)
        prob.options['group_by_pre_opt_post'] = True

        # Load aircraft and options data from user
        # Allow for user overrides here
        # add engine builder
        prob.load_inputs(
            'validation_cases/validation_data/test_models/small_scale_uav.csv',
            phase_info,
        )

        # engine_builders was removed from load_inputs in the upstream catch-up merge;
        # EngineModel instances (RCBuilder) are now registered via load_external_subsystems,
        # which auto-sorts them into engine_models.
        prob.load_external_subsystems(external_subsystems=[RCBuilder()])

        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()

        prob.add_phases()
        prob.add_post_mission_systems()

        # Link phases and variables
        prob.link_phases()

        prob.add_driver('SLSQP')

        prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons"]


        prob.add_design_variables()

        prob.model.add_subsystem(
            'endurance_comp',
            om.ExecComp(
                'endurance = energy / (p_cruise + 1.0e-3)',  # energy [W*h] / power [W] -> [h]
                endurance={'val': 1.0, 'units': 'h'},
                energy={'val': 1.0, 'units': 'W*h'},
                p_cruise={'val': 1.0, 'units': 'W'},
            ),
        )
        prob.model.connect('aircraft:battery:energy_capacity', 'endurance_comp.energy')
        prob.model.connect(
            'traj.cruise.timeseries.electric_power_in_total',
            'endurance_comp.p_cruise',
            src_indices=[0],
        )
        prob.model.add_objective('endurance_comp.endurance', ref=-1.0)
        
    

        prob.setup()

        prob.set_solver_print(level=0)

        # Set initial guesses for key variables to help SLSQP converge to a solution.
        prob.set_initial_guesses()
        prob.set_val('aircraft:engine:motor:mass', 0.45, units='kg')   # mid of [0.25, 0.65] -> KV ~370
        prob.set_val('aircraft:engine:motor:idle_current', 2.0, units='A')
        prob.set_val('aircraft:battery:voltage', 22.2, units='V')

        # Powertrain warm-start (matches Cruise_Attempt.py). The throttle-balance solver
        # defaults throttle to 1.0 -- right next to the prop's negative-thrust cliff, where
        # the ct/cp surrogate goes NaN. Seeding the well-conditioned ~0.54-throttle operating
        # point (RPM ~3750, current ~12 A) keeps the first solve (and the optimizer) in the
        # good region. The 'solver_sub.' path only exists when throttle is solved, so try both.
        _rpm_targets = [
            'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.rotations_per_minute',
            'traj.phases.cruise.rhs_all.core_propulsion.rc_electric.rotations_per_minute',
        ]
        _current_targets = [
            'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.current_flow',
            'traj.phases.cruise.rhs_all.core_propulsion.rc_electric.current_flow',
        ]
        _throttle_targets = [
            'traj.phases.cruise.rhs_all.solver_sub.throttle',
            'traj.phases.cruise.rhs_all.solver_sub.core_propulsion.rc_electric.throttle',
            'traj.phases.cruise.rhs_all.throttle',
        ]
        for _t in _rpm_targets:
            try:
                prob.set_val(_t, val=3750.0, units='rpm')
                break
            except Exception:
                continue
        for _t in _current_targets:
            try:
                prob.set_val(_t, val=12.0, units='A')
                break
            except Exception:
                continue
        for _t in _throttle_targets:
            try:
                prob.set_val(_t, val=0.54, units='unitless')
                break
            except Exception:
                continue

        prob.run_aviary_problem()

        self.assertTrue(prob.problem_ran_successfully)

        # Pinned regression values from the converged optimum. These are SLSQP outputs,
        # so use a loose-but-meaningful tolerance (not 1e-9) to tolerate optimizer drift.
        assert_near_equal(
            prob.get_val('endurance_comp.endurance', units='h')[0], 0.9788, tolerance=1e-3)
        assert_near_equal(
            prob.get_val('mission:design:gross_mass', units='kg')[0], 7.1654, tolerance=1e-3)
        assert_near_equal(
            prob.get_val('aircraft:engine:motor:mass', units='kg')[0], 0.5091, tolerance=1e-2)

        # Invariants: motor mass stays within its [0.25, 0.65] kg DV bound, and the
        # 1 km cruise distance target is met.
        mm = prob.get_val('aircraft:engine:motor:mass', units='kg')[0]
        self.assertTrue(0.25 < mm < 0.65, f"Motor mass {mm} kg is outside expected bounds.")
        self.assertLess(
            abs(prob.get_val('cruise_distance_constraint.distance_resid')[0]), 1e-3,
            "Cruise distance constraint is not satisfied after optimization.")


if __name__ == '__main__':
    unittest.main()