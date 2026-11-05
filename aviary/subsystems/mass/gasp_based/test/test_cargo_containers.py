import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs

from aviary.subsystems.mass.gasp_based.cargo_containers import CargoContainerMass
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft


@use_tempdirs
class CargoContainerMassTestCase(unittest.TestCase):
    """Tests for the CargoContainerMass component."""

    def _make_prob(self, num_passengers, uld_mass_per_passenger):
        options = AviaryValues()
        options.set_val(
            Aircraft.CrewPayload.Design.NUM_PASSENGERS, val=num_passengers, units='unitless'
        )
        options.set_val(
            Aircraft.CrewPayload.ULD_MASS_PER_PASSENGER, val=uld_mass_per_passenger, units='lbm'
        )

        prob = om.Problem()
        prob.model.add_subsystem(
            'cargo',
            CargoContainerMass(),
            promotes=['*'],
        )

        setup_model_options(prob, options)

        prob.setup(check=False, force_alloc_complex=True)
        return prob

    def test_large_single_aisle_1(self):
        """large_single_aisle_1_GASP.csv, generic_BWB_GASP ULD mass."""
        prob = self._make_prob(num_passengers=180, uld_mass_per_passenger=0)
        prob.run_model()

        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 165.0, 1e-7)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_bwb_parameters(self):
        """BWB parameters with a small passenger count."""
        prob = self._make_prob(num_passengers=5, uld_mass_per_passenger=0)
        prob.run_model()

        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 165.0, 1e-7)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    def test_nonzero_uld_mass_per_passenger(self):
        prob = self._make_prob(num_passengers=180, uld_mass_per_passenger=0.11)
        prob.run_model()

        with self.subTest(check='value'):
            assert_near_equal(prob[Aircraft.CrewPayload.CARGO_CONTAINER_MASS], 3300.0, 1e-5)

        with self.subTest(check='partials'):
            partial_data = prob.check_partials(out_stream=None, method='cs')
            assert_check_partials(partial_data, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
