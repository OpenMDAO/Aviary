import csv
import unittest
from copy import deepcopy
from pathlib import Path

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.testing_utils import set_env_vars, use_tempdirs

from aviary.models.missions.height_energy_default import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.utils.develop_metadata import add_meta_data
from aviary.variable_info.variable_meta_data import CoreMetaData
import aviary.api as av


@use_tempdirs
class TestReports(unittest.TestCase):
    def setUp(self):
        om.clear_reports()
        _clear_problem_names()

    @set_env_vars(TESTFLO_RUNNING='0', OPENMDAO_REPORTS='timeseries_csv')
    def test_timeseries_report(self):
        local_phase_info = deepcopy(phase_info)
        self.prob = run_aviary(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv',
            local_phase_info,
            optimizer='SLSQP',
            max_iter=0,
        )

        expected_header = [
            'time (s)',
            'altitude (ft)',
            'altitude_rate (ft/s)',
            'distance (m)',
            'drag (lbf)',
            'electric_power_in_total (kW)',
            'fuel_flow_rate_negative_total (lbm/h)',
            'mach (unitless)',
            'mach_rate (1/s)',
            'mass (kg)',
            'specific_energy_rate_excess (m/s)',
            'throttle (unitless)',
            'thrust_net_total (lbf)',
            'velocity (m/s)',
        ]

        expected_rows = [
            [
                '0.0',
                '0.0',
                '8.333333333333337',
                '1.0',
                '21108.341035874902',
                '0.0',
                '-10492.721631142893',
                '0.2',
                '0.0001354166666666668',
                '79560.101698',
                '12.349371130670201',
                '0.5654905755095957',
                '28479.14295846102',
                '68.0522432380756',
            ]
        ]

        report_file_path = (
            Path(self.prob.get_reports_dir()).joinpath('mission_timeseries_data.csv').absolute()
        )
        with open(report_file_path, mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # Read the header row

            # Validate the header
            self.assertEqual(expected_header, header, 'CSV header does not match expected values')

            for expected_row, output_row in zip(expected_rows, csvreader):
                for expected_val, output_val in zip(expected_row, output_row):
                    self.assertAlmostEqual(
                        float(expected_val),
                        float(output_val),
                        places=7,
                        msg='CSV row value does not match expected value within tolerance',
                    )

    @set_env_vars(TESTFLO_RUNNING='0', OPENMDAO_REPORTS='check_input_report')
    def test_check_input_report(self):
        # Make sure the input check works with custom metadata.
        # Make sure it also works when a user forgets to create metadata.

        class ExtraBuilder(SubsystemBuilder):
            def build_pre_mission(self, aviary_inputs):
                comp = om.ExecComp(['z = 2*x', 'p = q'])
                wing_group = om.Group()
                wing_group.add_subsystem(
                    'aerostructures',
                    comp,
                    promotes_inputs=[
                        ('x', 'aircraft:custom_var'),
                        ('q', 'aircraft:forgotten_input'),
                    ],
                    promotes_outputs=[('p', 'aircraft:forgotten_out')],
                )
                return wing_group

        metadata = deepcopy(CoreMetaData)
        local_phase_info = deepcopy(phase_info)

        local_phase_info['pre_mission']['external_subsystems'] = [ExtraBuilder()]

        add_meta_data(
            'aircraft:custom_var',
            metadata,
            units=None,
            desc='testing',
        )

        prob = AviaryProblem()
        prob.load_inputs(
            'models/aircraft/test_aircraft/aircraft_for_bench_FwFm.csv',
            local_phase_info,
        )

        prob.check_and_preprocess_inputs()

        prob.meta_data = metadata

        prob.build_model()

        prob.setup()

        # no need to run this model, just generate the report.
        prob.final_setup()

    @set_env_vars(TESTFLO_RUNNING='0')
    def test_multiple_off_design_report_directories(self):
        prob = av.AviaryProblem(verbosity=0)
        prob.load_inputs(
            'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv', phase_info
        )
        prob.check_and_preprocess_inputs()
        prob.build_model()
        prob.add_driver('SLSQP', max_iter=50)
        prob.add_design_variables()
        prob.add_objective()
        prob.setup()
        prob.run_aviary_problem()
        prob.run_off_design_mission(problem_type='fallout', mission_gross_mass=115000)
        prob.run_off_design_mission(problem_type='alternate', mission_range=1250)
        assert Path('testflo_off_design_1_out').is_dir()
        assert Path('testflo_off_design_out').is_dir()


if __name__ == '__main__':
    unittest.main()
