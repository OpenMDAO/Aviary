from copy import deepcopy
from pathlib import Path
import unittest

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.testing_utils import use_tempdirs

from aviary.core.aviary_problem import AviaryProblem
from aviary.models.missions.energy_state_default import phase_info

@use_tempdirs
class SimpleDragTest(unittest.TestCase):
    def setUp(self):
        om.clear_reports()
        _clear_problem_names()

    def test_transport(self):
        prob = AviaryProblem(name='p1', verbosity=0)

        local_phase_info = deepcopy(phase_info)
        local_phase_info['pre_mission']['aerodynamics'] = {
            'flops_report': True,
        }

        prob.load_inputs(
            'validation_cases/validation_data/test_models/aircraft_for_bench_FwFm.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_design_variables()
        prob.add_objective(objective_type='fuel_burned')
        prob.setup()
        prob.set_solver_print(0)

        # Subsystem reports generate after run.
        prob.run_model()

        report_path = Path(prob.get_reports_dir()) / 'subsystems' / 'aerodynamics.md'

        import os
        data=''
        for path in Path(os.getcwd()).rglob('*'):
            data += str(path)
        raise RuntimeError(data)

        with open(report_path, mode='r') as aerofile:
            lines = aerofile.readlines()

        # Just make sure the table data is there without drilling into each value.
        self.assertTrue('AERODYNAMIC DATA' in lines[0])
        self.assertTrue('SPAN EFFICIENCY FACTOR' in lines[2])
        self.assertTrue('PRESSURE DRAG' in lines[-15])

    def test_bwb(self):
        prob = AviaryProblem(verbosity=0)

        local_phase_info = deepcopy(phase_info)
        local_phase_info['pre_mission']['aerodynamics'] = {
            'flops_report': True,
        }

        prob.load_inputs(
            'models/aircraft/blended_wing_body/bwb_simple_FLOPS.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_design_variables()
        prob.add_objective(objective_type='fuel_burned')
        prob.setup()
        prob.set_solver_print(0)

        report_path = Path(prob.get_reports_dir()) / 'subsystems' / 'aerodynamics.md'

        # Subsystem reports generate after run.
        prob.run_model()

        with open(report_path, mode='r') as aerofile:
            lines = aerofile.readlines()

        # Just make sure the table data is there without drilling into each value.
        self.assertTrue('AERODYNAMIC DATA' in lines[0])
        self.assertTrue('SPAN EFFICIENCY FACTOR' in lines[2])
        self.assertTrue('PRESSURE DRAG' in lines[-15])


if __name__ == '__main__':
    unittest.main()
