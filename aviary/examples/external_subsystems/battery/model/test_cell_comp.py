from aviary.examples.external_subsystems.battery.model.cell_comp import CellComp
from openmdao.api import Problem, Group
from openmdao.utils.testing_utils import use_tempdirs
import unittest


class Test_cell_comp(unittest.TestCase):
    """
    test partials in CellComp component
    """

    @use_tempdirs
    def test_cell_comp(self):
        p = Problem()
        p.model = Group()
        p.model.add_subsystem('CellComp', CellComp(num_nodes=1), promotes=['*'])
        p.setup(mode='auto', check=True, force_alloc_complex=True)

        p.check_partials(compact_print=True, method='cs', step=1e-50)


if __name__ == "__main__":
    unittest.main()
