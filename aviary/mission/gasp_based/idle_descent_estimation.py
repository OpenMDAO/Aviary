import warnings

import openmdao.api as om

from aviary.interface.default_phase_info.gasp_fiti import create_gasp_based_descent_phases
from aviary.mission.gasp_based.phases.time_integration_traj import FlexibleTraj
from aviary.variable_info.variables import Aircraft, Mission, Dynamic


def descent_range_and_fuel(
    phases=None,
    ode_args=None,
    initial_mass=154e3,
    cruise_alt=35e3,
    cruise_mach=.8,
    empty_weight=85e3,
    payload_weight=30800,
    reserve_fuel=4998,
):

    prob = om.Problem()
    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = 'IPOPT'
    prob.driver.opt_settings['tol'] = 1.0E-6
    prob.driver.opt_settings['mu_init'] = 1e-5
    prob.driver.opt_settings['max_iter'] = 50
    prob.driver.opt_settings['print_level'] = 5

    if phases is None:
        phases = create_gasp_based_descent_phases(
            ode_args,
            cruise_mach=cruise_mach,
        )

    traj = FlexibleTraj(
        Phases=phases,
        traj_final_state_output=[Dynamic.Mission.MASS,
                                 Dynamic.Mission.DISTANCE, Dynamic.Mission.ALTITUDE],
        traj_initial_state_input=[
            Dynamic.Mission.MASS,
            Dynamic.Mission.DISTANCE,
            Dynamic.Mission.ALTITUDE,
        ],
    )
    prob.model = om.Group()
    prob.model.add_subsystem('traj', traj)

    prob.model.add_subsystem(
        "fuel_obj",
        om.ExecComp(
            "reg_objective = overall_fuel/10000",
            reg_objective={"val": 0.0, "units": "unitless"},
            overall_fuel={"units": "lbm"},
        ),
        promotes_inputs=[
            ("overall_fuel", Mission.Summary.TOTAL_FUEL_MASS),
        ],
        promotes_outputs=[("reg_objective", Mission.Objectives.FUEL)],
    )

    prob.model.add_objective(Mission.Objectives.FUEL, ref=1e4)

    prob.setup()
    prob.set_val("traj.altitude_initial", val=cruise_alt, units="ft")
    prob.set_val("traj.mass_initial", val=initial_mass, units="lbm")
    prob.set_val("traj.distance_initial", val=0, units="NM")

    # prevent UserWarning that is displayed when an event is triggered
    warnings.filterwarnings('ignore', category=UserWarning)
    prob.run_model()
    warnings.filterwarnings('default', category=UserWarning)

    final_range = prob.get_val('traj.distance_final', units='NM')[0]
    final_mass = prob.get_val('traj.mass_final', units='lbm')[0]
    descent_fuel = initial_mass - final_mass
    print('final range: ', final_range)
    print('fuel burned: ', descent_fuel)

    initial_mass2 = empty_weight + payload_weight + reserve_fuel + descent_fuel
    prob.set_val("traj.mass_initial", val=initial_mass2, units="lbm")

    # prevent UserWarning that is displayed when an event is triggered
    warnings.filterwarnings('ignore', category=UserWarning)
    prob.run_model()
    warnings.filterwarnings('default', category=UserWarning)

    final_range2 = prob.get_val('traj.distance_final', units='NM')[0]
    final_mass2 = prob.get_val('traj.mass_final', units='lbm')[0]
    descent_fuel2 = initial_mass2 - final_mass2
    print('initial mass: ', initial_mass2)
    print('final range: ', final_range2)
    print('fuel burned: ', initial_mass2-final_mass2)

    results = {
        'initial_guess': {
            'distance_flown': final_range,
            'fuel_burned': descent_fuel
        },
        'refined_guess': {
            'distance_flown': final_range2,
            'fuel_burned': descent_fuel2
        }
    }

    return results
