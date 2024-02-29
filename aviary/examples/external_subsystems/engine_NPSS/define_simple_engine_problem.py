"""
Define an aviary mission with an NPSS defined engine. During pre-mission the engine is designed and an engine deck is made.  
During the mission the deck is used for performance. Weight is estimated using the default Aviary method.
The engine model was developed using NPSS v3.2.
"""
from copy import deepcopy
import aviary.api as av

from aviary.examples.external_subsystems.engine_NPSS.table_engine_builder import TableEngineBuilder as EngineBuilder


def define_aviary_NPSS_problem():
    phase_info = deepcopy(av.default_height_energy_phase_info)

    prob = av.AviaryProblem()

    prob.options["group_by_pre_opt_post"] = True

    # Load aircraft and options data from user
    # Allow for user overrides here
    # add engine builder
    prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv',
                     phase_info, engine_builder=EngineBuilder())

    prob.add_pre_mission_systems()

    prob.add_phases()

    prob.add_post_mission_systems()

    # Link phases and variables
    prob.link_phases()

    prob.add_driver("SLSQP")

    prob.add_design_variables()

    prob.add_objective()

    prob.setup()

    prob.set_initial_guesses()
    return prob


if __name__ == "__main__":
    prob = define_aviary_NPSS_problem()
    prob.run_aviary_problem(suppress_solver_print=True)
    # print(prob.model.list_outputs())
    print('Design Mass Flow ='+str(prob['aircraft:engine:design_mass_flow'][0]))
    print('SLS scaled Thrust ='+str(prob['aircraft:engine:scaled_sls_thrust'][0]))

    print('Cruise neg fuel flow rate =' +
          str(prob['traj.cruise.rhs_all.NPSS_prop_system.fuel_flow_rate_negative'][0]))
    print('Net Cruise thrust =' +
          str(prob['traj.cruise.rhs_all.NPSS_prop_system.thrust_net'][0]))
    print('done')
