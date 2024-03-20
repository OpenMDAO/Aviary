from copy import deepcopy
from pathlib import Path
import unittest
import warnings
import csv
from openmdao.utils.testing_utils import use_tempdirs, set_env_vars
import os

from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.interface.methods_for_level1 import run_aviary
from aviary.variable_info.enums import Verbosity


@unittest.skip("Skipping test due to sensitivity in setup. Need tolerances on actual values")
class BasicReportTestCase(unittest.TestCase):
    def setUp(self):
        local_phase_info = deepcopy(phase_info)
        self.prob = run_aviary('models/test_aircraft/aircraft_for_bench_FwFm.csv',
                               local_phase_info, verbosity=Verbosity.QUIET,
                               mission_method="FLOPS", mass_method="FLOPS", optimizer='IPOPT')

    def test_text_report(self):
        self.prob.write_report('test_output.txt')

        expected_text = ['',
                         '',
                         'PROPULSION',
                         'aircraft:engine:scale_factor, 1.0 unitless',
                         'aircraft:propulsion:total_scaled_sls_thrust, 57856.2 lbf',
                         '',
                         'GEOMETRY: FLOPS METHOD',
                         'aircraft:wing:area, 1370.0 ft**2',
                         'aircraft:wing:span, 117.83 ft',
                         'aircraft:wing:aspect_ratio, 11.22091 unitless',
                         'aircraft:wing:sweep, 25.0 deg',
                         'aircraft:horizontal_tail:area, 355.0 ft**2',
                         'aircraft:vertical_tail:area, 284.0 ft**2',
                         'aircraft:fuselage:length, 128.0 ft',
                         'aircraft:fuselage:avg_diameter, 12.75 ft',
                         '',
                         'MASS ESTIMATION: FLOPS REGRESSIONS',
                         'aircraft:wing:mass, 17758.544900084722 lbm',
                         'aircraft:horizontal_tail:mass, 1817.6584389129562 lbm',
                         'aircraft:vertical_tail:mass, 1208.6209771133388 lbm',
                         'aircraft:fins:mass, 0.0 lbm',
                         'aircraft:canard:mass, 0.0 lbm',
                         'aircraft:fuselage:mass, 18357.133455139123 lbm',
                         'aircraft:landing_gear:total_mass, 0.0 lbm',
                         'aircraft:nacelle:mass, 1971.3819954084274 lbm',
                         'aircraft:design:structure_mass, 50200.36816077931 lbm',
                         'aircraft:propulsion:total_engine_mass, 14800.0 lbm',
                         'aircraft:propulsion:total_thrust_reversers_mass, 0.0 lbm',
                         'aircraft:propulsion:total_misc_mass, 648.8373506955577 lbm',
                         'aircraft:fuel:fuel_system_mass, 669.5772386254444 lbm',
                         'aircraft:propulsion:mass, 16118.414589321002 lbm',
                         'aircraft:controls:total_mass, 0.0 lbm',
                         'aircraft:apu:mass, 1142.065041407156 lbm',
                         'aircraft:instruments:mass, 601.1649288403382 lbm',
                         'aircraft:hydraulics:mass, 1086.6955064058336 lbm',
                         'aircraft:electrical:mass, 2463.8713804682375 lbm',
                         'aircraft:avionics:mass, 1652.6486922143051 lbm',
                         'aircraft:furnishings:mass, 15517.315000000002 lbm',
                         'aircraft:air_conditioning:mass, 1601.886853976589 lbm',
                         'aircraft:anti_icing:mass, 208.8500201913504 lbm',
                         'aircraft:design:systems_equip_mass, 25158.285516432577 lbm',
                         'aircraft:design:external_subsystems_mass, 0.0 lbm',
                         'aircraft:design:empty_mass, 91477.0682665329 lbm',
                         'aircraft:crew_and_payload:flight_crew_mass, 450.0 lbm',
                         'aircraft:crew_and_payload:non_flight_crew_mass, 465.0 lbm',
                         'aircraft:fuel:unusable_fuel_mass, 501.30242136015147 lbm',
                         'aircraft:propulsion:total_engine_oil_mass, 130.22722598615468 lbm',
                         'aircraft:crew_and_payload:passenger_service_mass, 3022.748058091815 lbm',
                         'aircraft:crew_and_payload:cargo_container_mass, 1400.00000001059 lbm',
                         'aircraft:design:operating_mass, 97446.3459719816 lbm',
                         'aircraft:crew_and_payload:passenger_mass, 30420.0 lbm',
                         'aircraft:crew_and_payload:passenger_payload_mass, 37856.0 lbm',
                         'aircraft:crew_and_payload:cargo_mass, 0.0 lbm',
                         'aircraft:design:zero_fuel_mass, 135302.3459719816 lbm',
                         'mission:design:fuel_mass, 39469.534814306564 lbm',
                         'mission:summary:total_fuel_mass, 39469.53476849914 lbm',
                         'mission:summary:gross_mass, 0.0 lbm',
                         '',
                         '']

        with open('test_output.txt',
                  'r') as f:
            text = f.readlines()

        for index, line in enumerate(text):
            if index < 1:
                # skip first lines (timestamp)
                continue

            expected_line = ''.join(expected_text[index].split())
            line_no_whitespace = ''.join(line.split())

            # Assert that the lines are equal
            try:
                self.assertEqual(line_no_whitespace.count(expected_line), 1)
            except Exception as error:
                exc_string = f'\nFound:    {line}\nExpected: {expected_text[index]}'
                raise Exception(exc_string)

    def test_file_rename(self):
        # ensure file exists
        with open(Path.cwd() / 'test_output.txt', 'w') as f:
            pass

        # catch warnings as errors
        warnings.filterwarnings("error")

        try:
            self.prob.write_report('test_output.txt')
        except UserWarning:
            pass
        else:
            self.fail('No warning that file output is moved to new name')

        # disable warnings as errors behavior for future tests
        warnings.resetwarnings()


@use_tempdirs
class AviaryMissionTestCase(unittest.TestCase):
    @set_env_vars(TESTFLO_RUNNING='0', OPENMDAO_REPORTS='mission')
    def test_mission_report(self):
        local_phase_info = deepcopy(phase_info)
        self.prob = run_aviary('models/test_aircraft/aircraft_for_bench_FwFm.csv',
                               local_phase_info,
                               optimizer='IPOPT',
                               max_iter=0)

        # Ensure that the mission_summary.md file exists
        mission_summary_file = Path(self.prob.get_reports_dir()
                                    ).joinpath('mission_summary.md')
        self.assertTrue(mission_summary_file.exists(),
                        "Mission summary file does not exist")

        # Ensure that the mission_summary.md file is not empty
        self.assertTrue(mission_summary_file.stat().st_size >
                        0, "Mission summary file is empty")

        # Ensure that the mission_summary.md file contains the expected header
        with open(mission_summary_file, 'r') as f:
            first_line = f.readline().strip()
            self.assertEqual(first_line, "# MISSION SUMMARY",
                             "Mission summary file does not contain expected header")

        # Define the phases and expected "Elapsed Time" values
        expected_phases = {
            "climb": "65.28",
            "cruise": "57.63",
            "descent": "29.58",
        }

        with open(mission_summary_file, 'r') as mdfile:
            content = mdfile.readlines()

        phase_found = {phase: False for phase in expected_phases}
        current_phase = None
        for line in content:
            if any(phase in line for phase in expected_phases):
                current_phase = [phase for phase in expected_phases if phase in line][0]
                phase_found[current_phase] = True
            elif line.startswith("| Elapsed Time |") and current_phase:
                # Extract the value from the table row
                elapsed_time_value = line.split("|")[2].strip()
                self.assertEqual(elapsed_time_value, expected_phases[current_phase],
                                 f"Elapsed Time for {current_phase} does not match expected value")
                current_phase = None  # Reset current_phase to ensure it's only set when a new phase is found

        # Ensure all phases were found
        for phase, found in phase_found.items():
            self.assertTrue(found, f"Phase {phase} not found in mission summary")

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
