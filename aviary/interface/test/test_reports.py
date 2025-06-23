import csv
import unittest
from copy import deepcopy
from pathlib import Path

import openmdao.api as om
from openmdao.core.problem import _clear_problem_names
from openmdao.utils.testing_utils import set_env_vars, use_tempdirs

from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.develop_metadata import add_meta_data
from aviary.variable_info.variable_meta_data import CoreMetaData


@use_tempdirs
class TestReports(unittest.TestCase):
    def setUp(self):
        om.clear_reports()
        _clear_problem_names()

    @set_env_vars(TESTFLO_RUNNING='0', OPENMDAO_REPORTS='timeseries_csv')
    def test_timeseries_report(self):
        local_phase_info = deepcopy(phase_info)
        self.prob = run_aviary(
            'models/test_aircraft/aircraft_for_bench_FwFm.csv',
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
                '21108.418300418845',
                '0.0',
                '-10492.593707324704',
                '0.2',
                '0.0001354166666666668',
                '79560.101698',
                '12.350271989430475',
                '0.565484286063171',
                '28478.788920867584',
                '68.05737270077049',
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

        class ExtraBuilder(SubsystemBuilderBase):
            def build_pre_mission(self, aviary_inputs):
                comp = om.ExecComp('z = 2*x')
                wing_group = om.Group()
                wing_group.add_subsystem(
                    'aerostructures',
                    comp,
                    promotes_inputs=[('x', 'aircraft:custom_var')],
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
            'models/test_aircraft/aircraft_for_bench_FwFm.csv',
            local_phase_info,
        )
        prob.check_and_preprocess_inputs()
        prob.meta_data = metadata

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        # no need to run this model, just generate the report.
        prob.final_setup()


if __name__ == '__main__':
    unittest.main()
