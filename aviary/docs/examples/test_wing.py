import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.docs.examples.wing import WingMassAndCOG

class WingMassTestCase(unittest.TestCase):
    """
    Wing mass test case

    """

    def setup(self):

        self.prob = om.Problem()
        self.prob.model.add_subsystem("wing", WingMassAndCOG(), promotes=["*"])