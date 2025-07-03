import subprocess
import unittest
from pathlib import Path

from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.utils.functions import get_aviary_resource_path


@use_tempdirs
class CommandEntryPointsTestCases(unittest.TestCase):
    def run_and_test_cmd(self, cmd):
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all. The underlying functions that implement the
        # commands should be tested separately.
        try:
            subprocess.check_output(cmd.split())
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode}")

    def get_file(self, filename):
        filepath = get_aviary_resource_path(filename)
        if not Path(filepath).exists():
            self.skipTest(f"couldn't find {filepath}")
        return filepath


class InstallationTest(CommandEntryPointsTestCases):
    def run_installation_test(self):
        cmd = 'aviary check'
        self.run_and_test_cmd(cmd)


class run_missionTestCases(CommandEntryPointsTestCases):
    @require_pyoptsparse(optimizer='SNOPT')
    def bench_test_SNOPT_cmd(self):
        cmd = 'aviary run_mission models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv --optimizer SNOPT --max_iter 1'
        self.run_and_test_cmd(cmd)

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_IPOPT_cmd(self):
        cmd = 'aviary run_mission models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv --optimizer IPOPT --max_iter 1'
        self.run_and_test_cmd(cmd)

    @require_pyoptsparse(optimizer='IPOPT')
    def bench_test_phase_info_cmd(self):
        cmd = (
            'aviary run_mission models/aircraft/test_aircraft/aircraft_for_bench_GwGm.csv --optimizer IPOPT --max_iter 1'
            ' --phase_info models/missions/two_dof_default.py'
        )
        self.run_and_test_cmd(cmd)


class fortran_to_aviaryTestCases(CommandEntryPointsTestCases):
    def test_diff_configuration_conversion(self):
        filepath = get_aviary_resource_path('utils/test/data/configuration_test_data_GASP.dat')
        outfile = Path.cwd() / 'utils/test/data/configuration_test_data_GASP' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd)

    def test_small_single_aisle_conversion(self):
        filepath = get_aviary_resource_path(
            'models/aircraft/small_single_aisle/small_single_aisle_GASP.dat'
        )
        outfile = Path.cwd() / 'small_single_aisle' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd)

    def test_FLOPS_conversion(self):
        filepath = get_aviary_resource_path(
            'models/aircraft/advanced_single_aisle/N3CC_generic_low_speed_polars_FLOPS.txt'
        )
        outfile = Path.cwd() / 'advanced_single_aisle' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l FLOPS'
        self.run_and_test_cmd(cmd)

    def test_force_conversion(self):
        filepath = get_aviary_resource_path(
            'models/aircraft/small_single_aisle/small_single_aisle_GASP.dat'
        )
        outfile = Path.cwd() / 'output.dat'
        cmd1 = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd1)
        filepath = get_aviary_resource_path(
            'models/aircraft/small_single_aisle/small_single_aisle_GASP.dat'
        )
        cmd2 = f'aviary fortran_to_aviary {filepath} -o {outfile} --force -l GASP'
        self.run_and_test_cmd(cmd2)


class hangarTestCases(CommandEntryPointsTestCases):
    def test_copy_folder(self):
        cmd = 'aviary hangar engines'
        self.run_and_test_cmd(cmd)

    def test_copy_deck(self):
        cmd = 'aviary hangar turbofan_22k.csv'
        self.run_and_test_cmd(cmd)

    def test_copy_n3cc_data(self):
        cmd = 'aviary hangar advanced_single_aisle/advanced_single_aisle_data.py'
        self.run_and_test_cmd(cmd)

    def test_copy_multiple(self):
        cmd = 'aviary hangar small_single_aisle_GASP.dat small_single_aisle_GASP.csv'
        self.run_and_test_cmd(cmd)

    def test_copy_to(self):
        outfile = Path.cwd() / 'example_files'
        cmd = f'aviary hangar small_single_aisle_GASP.dat -o {outfile}'
        self.run_and_test_cmd(cmd)


class convert_engineTestCases(CommandEntryPointsTestCases):
    def test_GASP_conversion(self):
        filepath = self.get_file('utils/test/data/GASP_turbofan_23k_1.eng')
        outfile = Path.cwd() / 'turbofan_23k_1.csv'
        cmd = f'aviary convert_engine {filepath} {outfile} -f GASP'
        self.run_and_test_cmd(cmd)

    def test_FLOPS_conversion(self):
        filepath = self.get_file('utils/test/data/FLOPS_turbofan_22k.txt')
        outfile = Path.cwd() / 'turbofan_22k.csv'
        cmd = f'aviary convert_engine {filepath} {outfile} -f FLOPS'
        self.run_and_test_cmd(cmd)

    def test_GASP_TS_conversion(self):
        filepath = self.get_file('models/engines/turboshaft_4465hp.eng')
        outfile = Path.cwd() / 'turboshaft_4465hp.eng'
        cmd = f'aviary convert_engine {filepath} {outfile} --data_format GASP_TS'
        self.run_and_test_cmd(cmd)


class convert_aero_tableTestCases(CommandEntryPointsTestCases):
    def test_GASP_conversion(self):
        filepath = self.get_file('subsystems/aerodynamics/gasp_based/data/GASP_aero_flaps.txt')
        outfile = Path.cwd() / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd)

    def test_FLOPS_conversion(self):
        filepath = self.get_file(
            'models/aircraft/advanced_single_aisle/N3CC_generic_low_speed_polars_FLOPSinp.txt'
        )
        outfile = Path.cwd() / 'advanced_single_aisle' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l FLOPS'
        self.run_and_test_cmd(cmd)


class convert_propeller_tableTestCases(CommandEntryPointsTestCases):
    """aviary convert_prop_table test. The only option is from GASP propeller map to Aviary table."""

    def test_GASP_conversion(self):
        filepath = self.get_file('models/propellers/PropFan.csv')
        outfile = Path.cwd() / 'output.dat'
        cmd = f'aviary convert_prop_table {filepath} {outfile} -f GASP'
        self.run_and_test_cmd(cmd)


if __name__ == '__main__':
    unittest.main()
