from copy import deepcopy
from pathlib import Path
import unittest
import csv
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars
import openmdao.api as om

from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.interface.methods_for_level1 import run_aviary


@use_tempdirs
class AviaryMissionTimeseries(unittest.TestCase):
    def setUp(self):
        om.clear_reports()

    @set_env_vars(TESTFLO_RUNNING='0', OPENMDAO_REPORTS='timeseries_csv')
    def test_timeseries_report(self):
        local_phase_info = deepcopy(phase_info)
        self.prob = run_aviary('models/test_aircraft/aircraft_for_bench_FwFm.csv',
                               local_phase_info,
                               optimizer='IPOPT',
                               max_iter=0)

        # Expected header names and units
        expected_header = ["time (s)", "mach (unitless)", "thrust_net_total (lbf)", "drag (lbf)",
                           "specific_energy_rate_excess (m/s)", "fuel_flow_rate_negative_total (lbm/h)",
                           "altitude_rate (ft/s)", "throttle (unitless)", "velocity (m/s)", "time_phase (s)",
                           "mach_rate (unitless/s)", "altitude (ft)", "mass (kg)", "distance (m)"]

        # Expected values for the first row of output
        expected_rows = [
            ["0.0", "0.2", "28334.271991561804", "21108.418300418845", "12.350271989430475", "-10440.375644236545", "8.16993480071768",
                "0.5629169556406933", "68.05737270077049", "0.0", "0.00013276144051166237", "0.0", "79560.101698", "1.0"],
        ]

        report_file_path = Path(self.prob.get_reports_dir()).joinpath(
            'mission_timeseries_data.csv').absolute()
        with open(report_file_path, mode='r') as csvfile:
            csvreader = csv.reader(csvfile)
            header = next(csvreader)  # Read the header row

            # Validate the header
            self.assertEqual(expected_header, header,
                             "CSV header does not match expected values")

            for expected_row, output_row in zip(expected_rows, csvreader):
                for expected_val, output_val in zip(expected_row, output_row):
                    self.assertAlmostEqual(float(expected_val), float(
                        output_val), places=7, msg="CSV row value does not match expected value within tolerance")


if __name__ == "__main__":
    unittest.main()
