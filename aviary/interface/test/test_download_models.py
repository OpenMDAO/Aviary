import subprocess
import unittest
from pathlib import Path
import shutil

import pkg_resources
from openmdao.utils.testing_utils import use_tempdirs
from aviary.interface.download_models import get_model, save_file


@use_tempdirs
class CommandEntryPointsTestCases(unittest.TestCase):

    def run_and_test_hanger(self, filenames, out_dir=''):
        # this only tests that a given command line tool returns a 0 return code. It doesn't
        # check the expected output at all.  The underlying functions that implement the
        # commands should be tested seperately.
        if isinstance(filenames, str):
            filenames = [filenames]
        cmd = ['aviary', 'hanger'] + filenames

        if out_dir:
            cmd += ['-o', out_dir]
            out_dir = Path(out_dir)
        else:
            out_dir = Path.cwd() / 'aviary_models'

        try:
            output = subprocess.check_output(cmd)
            for filename in filenames:
                path = out_dir / filename.split('/')[-1]
                self.assertTrue(path.exists())
        except subprocess.CalledProcessError as err:
            self.fail(f"Command '{cmd}' failed.  Return code: {err.returncode}")

    def test_single_file_without_path(self):
        filename = 'turbofan_22k.deck'
        self.run_and_test_hanger(filename)

    def test_single_file_with_path(self):
        filename = 'engines/turbofan_22k.deck'
        self.run_and_test_hanger(filename)

    def test_multiple_files(self):
        filenames = ['small_single_aisle_GwGm.dat', 'small_single_aisle_GwGm.csv']
        self.run_and_test_hanger(filenames)

    def test_folder(self):
        filename = 'engines'
        self.run_and_test_hanger(filename)

    def test_single_file_custom_outdir(self):
        filename = 'small_single_aisle_GwGm.csv'
        out_dir = '~/test_hanger'
        self.run_and_test_hanger(filename, out_dir)
        shutil.rmtree(out_dir)

    def test_expected_path(self):
        filename = f'test_aircraft/converter_configuration_test_data_GwGm.dat'
        aviary_path = get_model('converter_configuration_test_data_GwGm.dat')

        expected_path = pkg_resources.resource_filename('aviary',
                                                        'models/test_aircraft/converter_configuration_test_data_GwGm.dat')
        self.assertTrue(str(aviary_path) == expected_path)


if __name__ == "__main__":
    unittest.main()
