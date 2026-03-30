import unittest

from aviary.interface.installation_test import _exec_installation_test


class test_installation(unittest.TestCase):
    def test_installation(self):
        success = _exec_installation_test(None, None)
        assert success


if __name__ == '__main__':
    unittest.main()
