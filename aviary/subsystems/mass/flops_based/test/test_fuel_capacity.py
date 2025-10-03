import unittest

import openmdao.api as om
from parameterized import parameterized

from aviary.subsystems.mass.flops_based.fuel_capacity import (
    AuxFuelCapacity,
    FuelCapacityGroup,
    FuselageFuelCapacity,
    TotalFuelCapacity,
    WingFuelCapacity,
)
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.test_utils.variable_test import assert_match_varnames
from aviary.validation_cases.validation_tests import (
    do_validation_test,
    flops_validation_test,
    get_flops_case_names,
    get_flops_inputs,
    print_case,
)
from aviary.variable_info.functions import override_aviary_vars
from aviary.variable_info.variables import Aircraft


class FuelCapacityGroupTest(unittest.TestCase):
    @parameterized.expand(get_flops_case_names(only=['AdvancedSingleAisle']), name_func=print_case)
    def test_case(self, case_name):
        class PreMission(om.Group):
            def initialize(self):
                self.options.declare(
                    'aviary_options', types=AviaryValues, desc='Aircraft options dictionary'
                )

            def setup(self):
                self.add_subsystem(
                    'fuel_capacity_group',
                    FuelCapacityGroup(),
                    promotes_inputs=['*'],
                    promotes_outputs=['*'],
                )

            def configure(self):
                aviary_options = self.options['aviary_options']

                # Overrides
                override_aviary_vars(self, aviary_options)

        prob = om.Problem()
        prob.model.add_subsystem(
            'premission',
            PreMission(aviary_options=get_flops_inputs(case_name)),
            promotes_outputs=['*'],
            promotes_inputs=['*'],
        )

        prob.setup(check=False, force_alloc_complex=True)

        flops_validation_test(
            prob,
            case_name,
            input_keys=[
                Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
                Aircraft.Fuel.WING_FUEL_FRACTION,
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD,
            ],
            output_keys=Aircraft.Fuel.TOTAL_CAPACITY,
            atol=1e-10,
            rtol=1e-10,
        )


wing_capacity_data = {}
wing_capacity_data['1'] = AviaryValues(
    {
        Aircraft.Fuel.DENSITY: (8.04, 'lbm/galUS'),
        Aircraft.Fuel.WING_REF_CAPACITY: (30.0, 'lbm'),
        Aircraft.Fuel.WING_REF_CAPACITY_AREA: (200.0, 'unitless'),
        Aircraft.Fuel.WING_REF_CAPACITY_TERM_B: (1.3, 'unitless'),
        Aircraft.Fuel.WING_FUEL_FRACTION: (0.7752, 'unitless'),
        Aircraft.Wing.AREA: (150.0, 'ft**2'),
        Aircraft.Wing.SPAN: (17.0, 'ft'),
        Aircraft.Wing.TAPER_RATIO: (1.5, 'unitless'),
        Aircraft.Wing.THICKNESS_TO_CHORD: (0.33, 'unitless'),
        Aircraft.Fuel.WING_REF_CAPACITY_TERM_A: (-100.0, 'unitless'),
    }
)
wing_capacity_data['2'] = AviaryValues(
    {
        Aircraft.Fuel.DENSITY: (8.04, 'lbm/galUS'),
        Aircraft.Fuel.WING_REF_CAPACITY: (30.0, 'lbm'),
        Aircraft.Fuel.WING_REF_CAPACITY_AREA: (200.0, 'unitless'),
        Aircraft.Fuel.WING_REF_CAPACITY_TERM_B: (1.3, 'unitless'),
        Aircraft.Fuel.WING_FUEL_FRACTION: (0.7752, 'unitless'),
        Aircraft.Wing.AREA: (150.0, 'ft**2'),
        Aircraft.Wing.SPAN: (17.0, 'ft'),
        Aircraft.Wing.TAPER_RATIO: (1.5, 'unitless'),
        Aircraft.Wing.THICKNESS_TO_CHORD: (0.33, 'unitless'),
        Aircraft.Fuel.WING_REF_CAPACITY_TERM_A: (1.2, 'unitless'),
    }
)

wing_capacity_cases = [key for key in wing_capacity_data]


class WingFuelCapacityTest(unittest.TestCase):
    @parameterized.expand(wing_capacity_cases, name_func=print_case)
    def test_derivs(self, case_name):
        validation_data = wing_capacity_data[case_name]
        prob = om.Problem()

        prob.model.add_subsystem('comp', WingFuelCapacity(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=validation_data,
            output_validation_data=validation_data,
            input_keys=[
                Aircraft.Fuel.DENSITY,
                Aircraft.Fuel.WING_REF_CAPACITY,
                Aircraft.Fuel.WING_REF_CAPACITY_AREA,
                Aircraft.Fuel.WING_REF_CAPACITY_TERM_B,
                Aircraft.Fuel.WING_FUEL_FRACTION,
                Aircraft.Wing.AREA,
                Aircraft.Wing.SPAN,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD,
                Aircraft.Fuel.WING_REF_CAPACITY_TERM_A,
            ],
            output_keys=Aircraft.Fuel.WING_FUEL_CAPACITY,
            atol=1e-10,
            # TODO: No wing fuel capacity validation data, check only partials
            check_values=False,
        )


fuse_capacity_data = {}
fuse_capacity_data['1'] = AviaryValues(
    {
        Aircraft.Fuel.TOTAL_CAPACITY: (100.0, 'lbm'),
        Aircraft.Fuel.WING_FUEL_CAPACITY: (73.0, 'lbm'),
        Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY: (27.0, 'lbm'),
    }
)

fuse_capacity_cases = [key for key in fuse_capacity_data]


class FuselageFuelCapacityTest(unittest.TestCase):
    @parameterized.expand(fuse_capacity_cases, name_func=print_case)
    def test_basic(self, case_name):
        validation_data = fuse_capacity_data[case_name]
        prob = om.Problem()

        prob.model.add_subsystem('fuel', FuselageFuelCapacity(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=validation_data,
            output_validation_data=validation_data,
            input_keys=[Aircraft.Fuel.TOTAL_CAPACITY, Aircraft.Fuel.WING_FUEL_CAPACITY],
            output_keys=Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
            tol=1.0e-10,
            atol=1e-10,
        )


aux_capacity_data = {}
aux_capacity_data['1'] = AviaryValues(
    {
        Aircraft.Fuel.TOTAL_CAPACITY: (100.0, 'lbm'),
        Aircraft.Fuel.WING_FUEL_CAPACITY: (25.0, 'lbm'),
        Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY: (33.0, 'lbm'),
        Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY: (42.0, 'lbm'),
    }
)

aux_capacity_cases = [key for key in aux_capacity_data]


class AuxFuelCapacityTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(aux_capacity_cases, name_func=print_case)
    def test_basic(self, case_name):
        validation_data = aux_capacity_data[case_name]
        prob = self.prob

        prob.model.add_subsystem('fuel', AuxFuelCapacity(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=validation_data,
            output_validation_data=validation_data,
            input_keys=[
                Aircraft.Fuel.TOTAL_CAPACITY,
                Aircraft.Fuel.WING_FUEL_CAPACITY,
                Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
            ],
            output_keys=Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
            tol=1.0e-10,
            atol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


total_capacity_data = {}
total_capacity_data['1'] = AviaryValues(
    {
        Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY: (100.0, 'lbm'),
        Aircraft.Fuel.WING_FUEL_CAPACITY: (25.0, 'lbm'),
        Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY: (33.0, 'lbm'),
        Aircraft.Fuel.TOTAL_CAPACITY: (158.0, 'lbm'),
    }
)

total_capacity_cases = [key for key in total_capacity_data]


class TotalFuelCapacityTest(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()

    @parameterized.expand(total_capacity_cases, name_func=print_case)
    def test_basic(self, case_name):
        validation_data = total_capacity_data[case_name]
        prob = self.prob

        prob.model.add_subsystem('fuel', TotalFuelCapacity(), promotes=['*'])

        prob.setup(force_alloc_complex=True)

        do_validation_test(
            prob,
            case_name,
            input_validation_data=validation_data,
            output_validation_data=validation_data,
            input_keys=[
                Aircraft.Fuel.AUXILIARY_FUEL_CAPACITY,
                Aircraft.Fuel.WING_FUEL_CAPACITY,
                Aircraft.Fuel.FUSELAGE_FUEL_CAPACITY,
            ],
            output_keys=Aircraft.Fuel.TOTAL_CAPACITY,
            tol=1.0e-10,
            atol=1e-10,
        )

    def test_IO(self):
        assert_match_varnames(self.prob.model)


if __name__ == '__main__':
    unittest.main()
