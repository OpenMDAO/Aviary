import shutil
import unittest
from pathlib import Path

from openmdao.utils.testing_utils import use_tempdirs

from aviary.interface.download_models import get_model, save_file
from aviary.utils.functions import get_aviary_resource_path


@use_tempdirs
class TestDownloadModels(unittest.TestCase):
    def run_and_test_hangar(self, filename, out_dir=''):
        # tests that the function runs successfully and that the files are generated
        if out_dir:
            out_dir = Path(out_dir)
        else:
            out_dir = Path.cwd() / 'aviary_models'

        path = get_model(filename)
        save_file(path, outdir=out_dir)
        path = out_dir / filename.split('/')[-1]
        self.assertTrue(path.exists())

    def test_single_file_without_path(self):
        filename = 'turbofan_22k.csv'
        self.run_and_test_hangar(filename)

    def test_single_file_with_path(self):
        filename = 'engines/turbofan_22k.csv'
        self.run_and_test_hangar(filename)

    def test_folder(self):
        filename = 'engines'
        self.run_and_test_hangar(filename)

    def test_single_file_custom_outdir(self):
        filename = 'small_single_aisle_GASP.csv'
        out_dir = 'test_hangar'
        self.run_and_test_hangar(filename, out_dir)
        shutil.rmtree(out_dir)

    def test_expected_path(self):
        aviary_path = get_model('large_single_aisle_1_GASP.dat')

        expected_path = get_aviary_resource_path(
            'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.dat'
        )
        self.assertTrue(str(aviary_path) == str(expected_path))


if __name__ == '__main__':
    unittest.main()
    # test = CommandEntryPointsTestCases()
    # test.test_single_file_with_path()
