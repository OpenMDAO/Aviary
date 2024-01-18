import dymos as dm
import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np

from aviary.mission.gasp_based.phases.accel_phase import AccelPhase
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.interface.default_phase_info.two_dof import default_mission_subsystems


def run_accel():
    # Column-wise data from CSV
    time_GASP = np.array([0.182, 0.188])
    distance_GASP = np.array([0.00, 1.57])
    weight_GASP = np.array([174974., 174878.])
    TAS_GASP = np.array([185., 252.])
    EAS_GASP = np.array([184., 250.])

    time_GASP = time_GASP * 3600
    time_GASP = time_GASP - time_GASP[0]

    prob = om.Problem(model=om.Group())

    prob.driver = om.pyOptSparseDriver()
    prob.driver.options["optimizer"] = "SNOPT"
    prob.driver.opt_settings["Major iterations limit"] = 100
    prob.driver.opt_settings["Major optimality tolerance"] = 5.0e-3
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-6
    prob.driver.opt_settings["iSumm"] = 6
    prob.driver.declare_coloring()

    transcription = dm.Radau(num_segments=15, order=4, compressed=False)

    aviary_options = get_option_defaults()

    phase_name = "accel"
    phase_options = {
        'user_options': {
            'alt': (500, 'ft'),
            'EAS_constraint_eq': (250, 'kn'),
            'fix_initial': True,
            'time_initial_bounds': ((0, 0), 's'),
            'duration_bounds': ((0, 36_000), 's'),
            'duration_ref': (1000, 's'),
            'TAS_lower': (0, 'kn'),
            'TAS_upper': (1000, 'kn'),
            'TAS_ref': (250, 'kn'),
            'TAS_ref0': (150, 'kn'),
            'mass_lower': (0, 'lbm'),
            'mass_upper': (None, 'lbm'),
            'mass_ref': (175_500, 'lbm'),
            'distance_lower': (0, 'NM'),
            'distance_upper': (150, 'NM'),
            'distance_ref': (1, 'NM'),
            'distance_defect_ref': (1, 'NM'),
        }
    }

    phase_object = AccelPhase.from_phase_info(
        phase_name, phase_options, default_mission_subsystems, transcription=transcription)
    accel = phase_object.build_phase(aviary_options=aviary_options)

    accel.add_parameter(
        Aircraft.Wing.AREA, opt=False, units="ft**2", val=1370.0, static_target=True
    )

    prob.model.add_subsystem(name='accel', subsys=accel)
    accel.add_timeseries_output("EAS", output_name="EAS", units="kn")
    accel.add_timeseries_output("TAS", output_name="TAS", units="kn")
    accel.add_timeseries_output("mass", output_name="mass", units="lbm")
    accel.add_timeseries_output(Dynamic.Mission.DISTANCE,
                                output_name=Dynamic.Mission.DISTANCE, units="NM")
    accel.timeseries_options['use_prefix'] = True

    accel.add_objective("time", loc="final")
    prob.model.linear_solver.options["iprint"] = 0
    prob.model.nonlinear_solver.options["iprint"] = 0

    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)

    prob.set_solver_print(level=0)

    prob.setup()
    prob.set_val("accel.states:mass", accel.interp("mass", ys=[174974, 174878]))
    prob.set_val("accel.states:TAS", accel.interp("TAS", ys=[185, 252]))
    prob.set_val("accel.states:distance", accel.interp(
        Dynamic.Mission.DISTANCE, ys=[0, 1.57]))
    prob.set_val("accel.t_duration", 1000)
    prob.set_val("accel.t_initial", 0)

    dm.run_problem(problem=prob, simulate=True)
    print()
    print("Done Running Model, Now Simulating")
    print()

    return prob
