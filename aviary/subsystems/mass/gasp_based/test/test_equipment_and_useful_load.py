import unittest
import os

import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.mass.gasp_based.equipment_and_useful_load import \
    EquipAndUsefulLoadMass
from aviary.variable_info.options import get_option_defaults
from aviary.utils.test_utils.IO_test_util import assert_match_spec, skipIfMissingXDSM
from aviary.variable_info.enums import GASPEngineType
from aviary.variable_info.variables import Aircraft, Mission


# this is the large single aisle 1 V3 test case
class FixedEquipMassTestCase1(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "equip",
            EquipAndUsefulLoadMass(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=[29500.0], units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 21089, tol)
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5176, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 4932

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase2(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=5, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "equip",
            EquipAndUsefulLoadMass(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 1235.4, tol)
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23509, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase3(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.Engine.TYPE,
                        val=[GASPEngineType.RECIP_CARB], units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "equip",
            EquipAndUsefulLoadMass(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 892.83, tol)
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23509, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase4smooth(
    unittest.TestCase
):  # this is the large single aisle 1 V3 test case
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "equip",
            EquipAndUsefulLoadMass(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4

        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 21089, tol)
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5176, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 4932

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase5smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=5, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "equip",
            EquipAndUsefulLoadMass(
                aviary_options=options,
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 1235.4, tol)
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23509, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class FixedEquipMassTestCase6smooth(unittest.TestCase):
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=5, units='unitless')
        options.set_val(Aircraft.Engine.TYPE,
                        val=[GASPEngineType.RECIP_CARB], units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')
        options.set_val(Aircraft.Design.SMOOTH_MASS_DISCONTINUITIES,
                        val=True, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "equip",
            EquipAndUsefulLoadMass(
                aviary_options=options
            ),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        # not actual GASP value
        assert_near_equal(self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 892.83, tol)
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 23509, tol
        )  # not actual GASP value

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)


class EquipAndUsefulMassGroupTestCase1(
    unittest.TestCase
):  # this is the large single aisle 1 V3 test case
    def setUp(self):

        options = get_option_defaults()
        options.set_val(Aircraft.CrewPayload.NUM_PASSENGERS, val=180, units='unitless')
        options.set_val(Aircraft.LandingGear.FIXED_GEAR,
                        val=False, units='unitless')

        self.prob = om.Problem()
        self.prob.model.add_subsystem(
            "group",
            EquipAndUsefulLoadMass(aviary_options=options),
            promotes=["*"],
        )

        self.prob.model.set_input_defaults(
            Aircraft.Design.EQUIPMENT_MASS_COEFFICIENTS,
            val=[
                928.0,
                0.0736,
                0.112,
                0.14,
                1959.0,
                1.65,
                551.0,
                11192.0,
                5.0,
                3.0,
                50.0,
                7.6,
                12.0,
            ],
            units="unitless",
        )
        self.prob.model.set_input_defaults(
            Aircraft.Wing.AREA, val=1370.3, units="ft**2"
        )

        self.prob.model.set_input_defaults(
            Mission.Design.GROSS_MASS, val=175400, units="lbm"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.LENGTH, val=129.4, units="ft"
        )
        self.prob.model.set_input_defaults(Aircraft.Wing.SPAN, val=117.8, units="ft")
        self.prob.model.set_input_defaults(
            Aircraft.LandingGear.TOTAL_MASS, val=7511, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.Controls.TOTAL_MASS, val=3895.0, units="lbm")
        self.prob.model.set_input_defaults(
            Aircraft.HorizontalTail.AREA, val=375.9, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.VerticalTail.AREA, val=469.3, units="ft**2"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.PRESSURE_DIFFERENTIAL, val=7.5, units="psi")
        self.prob.model.set_input_defaults(
            Aircraft.Fuselage.AVG_DIAMETER, val=13.1, units="ft")

        self.prob.model.set_input_defaults(
            Aircraft.Engine.SCALED_SLS_THRUST, val=29500.0, units="lbf"
        )
        self.prob.model.set_input_defaults(
            Aircraft.Fuel.WING_FUEL_FRACTION, val=0.6, units="unitless"
        )

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):

        self.prob.run_model()

        tol = 1e-4
        assert_near_equal(
            self.prob[Aircraft.Design.FIXED_USEFUL_LOAD], 5176, tol
        )  # modified from GASP value to account for updated crew mass. GASP value is 4932
        assert_near_equal(self.prob[Aircraft.Design.FIXED_EQUIPMENT_MASS], 21089, tol)

        partial_data = self.prob.check_partials(out_stream=None, method="cs")
        assert_check_partials(partial_data, atol=8e-12, rtol=1e-12)

    @skipIfMissingXDSM('mass_and_sizing_basic_specs/equip.json')
    def test_io_equip_and_useful_group_spec(self):

        subsystem = self.prob.model

        assert_match_spec(subsystem, "mass_and_sizing_basic_specs/equip.json")


if __name__ == "__main__":
    unittest.main()
