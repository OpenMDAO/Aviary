from aviary.api import AnalysisScheme
from aviary.examples.level2_shooting_traj import run_aviary
from aviary.interface.default_phase_info.two_dof import phase_info
from openmdao.utils.testing_utils import use_tempdirs
import unittest


@use_tempdirs
def test_run_aviary():
    input_deck = 'models/large_single_aisle_1/large_single_aisle_1_GwGm.csv'
    run_aviary(input_deck, phase_info,
               analysis_scheme=AnalysisScheme.SHOOTING, run_driver=False)


if __name__ == "__main__":
    unittest.main()
