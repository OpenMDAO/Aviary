import unittest

from openmdao.utils.testing_utils import use_tempdirs

from aviary.test_install import installation_test


class test_installation(unittest.TestCase):
    def test_installation(self):
        success = installation_test()
        assert success


if __name__ == '__main__':
    unittest.main()
