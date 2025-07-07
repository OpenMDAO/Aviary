import unittest

from openmdao.utils.testing_utils import require_pyoptsparse, use_tempdirs

from aviary.api import AnalysisScheme
from aviary.examples.run_level2_shooting_traj import custom_run_aviary


@unittest.skip(
    'Shooting method is not correctly receiving user-set options, and is currently '
    'using default values for most options'
)
@use_tempdirs
class CustomTrajTestCase(unittest.TestCase):
    # A test class for shooting scheme

    @require_pyoptsparse(optimizer='IPOPT')
    def test_run_aviary(self):
        self.skipTest('SGM currently failing this test.')
        input_deck = 'models/aircraft/large_single_aisle_1/large_single_aisle_1_GASP.csv'
        custom_run_aviary(input_deck, analysis_scheme=AnalysisScheme.SHOOTING, run_driver=False)


if __name__ == '__main__':
    unittest.main()
