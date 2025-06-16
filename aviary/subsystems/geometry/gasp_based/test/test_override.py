import unittest
import warnings

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.interface.methods_for_level2 import AviaryGroup
from aviary.subsystems.aerodynamics.gasp_based.gaspaero import AeroGeom
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.preprocessors import preprocess_propulsion
from aviary.utils.process_input_decks import create_vehicle
from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
from aviary.variable_info.variables import Aircraft


class GASPOverrideTestCase(unittest.TestCase):
    def setUp(self):
        aviary_inputs, initial_guesses = create_vehicle(
            'models/test_aircraft/configuration_test_GASP.csv'
        )

        engines = [build_engine_deck(aviary_inputs)]

        core_subsystems = get_default_premission_subsystems('GASP', engines)
        preprocess_propulsion(aviary_inputs, engines)

        self.aviary_inputs = aviary_inputs

        prob = om.Problem()

        aviary_options = aviary_inputs
        subsystems = core_subsystems

        prob.model = AviaryGroup(aviary_options=aviary_options, aviary_metadata=BaseMetaData)

        prob.model.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_options, subsystems=subsystems),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        self.prob = prob

    def test_case1(self):
        # Test override: expect the given value
        prob = self.prob

        self.aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA, val=4000.0, units='ft**2')

        setup_model_options(prob, self.aviary_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            prob.setup()

        prob.run_model()

        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, 1e-6)

    def test_case2(self):
        # Test override: expect the computed value
        prob = self.prob

        # self.aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA, val=4000, units="ft**2")

        setup_model_options(prob, self.aviary_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            prob.setup()

        prob.run_model()

        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4794.748, 1e-6)

    def test_case3(self):
        # Test WETTED_AREA_SCALER: expected half of the computed value
        prob = self.prob

        # self.aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA, val=4000, units="ft**2")
        self.aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.5, units='unitless')

        setup_model_options(prob, self.aviary_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            prob.setup()

        prob.run_model()

        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 2397.374, 1e-6)

    def test_case4(self):
        # Test WETTED_AREA_SCALER: expect no effect
        prob = self.prob

        self.aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA, val=4000, units='ft**2')
        self.aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA_SCALER, val=0.5, units='unitless')

        setup_model_options(prob, self.aviary_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            prob.setup()

        prob.run_model()

        assert_near_equal(self.prob[Aircraft.Fuselage.WETTED_AREA], 4000, 1e-6)

    def test_case_aero_coeffs(self):
        """
        Test overriding from csv (vertical tail) and overriding from code (horizontal tail)
        Also checks non-overriden (wing) and default (strut).
        """
        prob = self.prob
        prob.model.add_subsystem('geom', AeroGeom(), promotes=['*'])
        self.aviary_inputs.set_val(Aircraft.HorizontalTail.FORM_FACTOR, val=1.5)

        setup_model_options(prob, self.aviary_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            prob.setup()

        prob.run_model()

        assert_near_equal(self.prob[Aircraft.Wing.FORM_FACTOR], 2.47320154, 1e-6)
        assert_near_equal(self.prob[Aircraft.HorizontalTail.FORM_FACTOR], 1.5, 1e-6)
        assert_near_equal(self.prob[Aircraft.VerticalTail.FORM_FACTOR], 2, 1e-6)
        assert_near_equal(self.prob[Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR], 1.125, 1e-6)

    def test_case_emp_geo(self):
        # Make sure we can override horizontal and veritcal tail geo vars.
        prob = self.prob

        self.aviary_inputs.set_val(Aircraft.HorizontalTail.AREA, val=7777.0, units='ft**2')
        self.aviary_inputs.set_val(Aircraft.VerticalTail.SPAN, val=8888.0, units='ft')

        setup_model_options(prob, self.aviary_inputs)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', om.PromotionWarning)
            prob.setup()

        prob.run_model()

        assert_near_equal(
            self.prob.get_val(Aircraft.HorizontalTail.AREA, units='ft**2'),
            7777.0,
            1e-6,
        )
        assert_near_equal(
            self.prob.get_val(Aircraft.VerticalTail.SPAN, units='ft'),
            8888.0,
            1e-6,
        )


if __name__ == '__main__':
    # unittest.main()
    test = GASPOverrideTestCase()
    test.setUp()
    test.test_case_emp_geo()
