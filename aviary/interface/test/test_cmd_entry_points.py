import subprocess
import unittest
from pathlib import Path

import pkg_resources
from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs


@use_tempdirs
class CommandEntryPointsTestCases(unittest.TestCase):

    def run_and_test_cmd(self, cmd):
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        try:
            output = subprocess.check_output(cmd.split())
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode}")

    @require_pyoptsparse(optimizer="SNOPT")
    def bench_test_SNOPT_cmd(self):
        cmd = 'aviary run_mission models/test_aircraft/aircraft_for_bench_GwGm.csv --optimizer SNOPT --mass_origin GASP --mission_method GASP --max_iter 1'
        self.run_and_test_cmd(cmd)

    @require_pyoptsparse(optimizer="IPOPT")
    def bench_test_IPOPT_cmd(self):
        cmd = 'aviary run_mission models/test_aircraft/aircraft_for_bench_GwGm.csv --optimizer IPOPT --mass_origin GASP --mission_method GASP --max_iter 1'
        self.run_and_test_cmd(cmd)

    def test_diff_configuration_conversion(self):
        filepath = pkg_resources.resource_filename('aviary',
                                                   'models/test_aircraft/converter_configuration_test_data_GwGm.dat')
        outfile = Path.cwd() / 'test_aircraft/converter_configuration_test_data_GwGm' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd)

    def test_small_single_aisle_conversion(self):
        filepath = pkg_resources.resource_filename('aviary',
                                                   'models/small_single_aisle/small_single_aisle_GwGm.dat')
        outfile = Path.cwd() / 'small_single_aisle' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd)

    def test_FLOPS_conversion(self):
        filepath = pkg_resources.resource_filename('aviary',
                                                   'models/N3CC/N3CC_generic_low_speed_polars_FLOPSinp.txt')
        outfile = Path.cwd() / 'N3CC' / 'output.dat'
        cmd = f'aviary fortran_to_aviary {filepath} -o {outfile} -l FLOPS'
        self.run_and_test_cmd(cmd)

    def test_force_conversion(self):
        filepath = pkg_resources.resource_filename('aviary',
                                                   'models/test_aircraft/converter_configuration_test_data_GwGm.dat')
        outfile = Path.cwd() / 'output.dat'
        cmd1 = f'aviary fortran_to_aviary {filepath} -o {outfile} -l GASP'
        self.run_and_test_cmd(cmd1)
        filepath = pkg_resources.resource_filename('aviary',
                                                   'models/test_aircraft/converter_configuration_test_data_GwGm.dat')
        cmd2 = f'aviary fortran_to_aviary {filepath} -o {outfile} --force -l GASP'
        self.run_and_test_cmd(cmd2)


if __name__ == "__main__":
    unittest.main()
