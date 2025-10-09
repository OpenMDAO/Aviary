import unittest

import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal
from aviary.subsystems.mass.flops_based.wing_group import WingMassGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Mission, Settings


class BWBWingGroupTest(unittest.TestCase):
    def setUp(self):
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 1, units='unitless')
        aviary_options.set_val(Aircraft.Design.TYPE, val='BWB', units='unitless')
        aviary_options.set_val(Aircraft.Wing.DETAILED_WING, val=True, units='unitless')
        aviary_options.set_val(Aircraft.Engine.NUM_ENGINES, [3], units='unitless')
        aviary_options.set_val(Aircraft.Engine.NUM_WING_ENGINES, [0], units='unitless')
        aviary_options.set_val(Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 0, units='unitless')
        aviary_options.set_val(
            Aircraft.Wing.INPUT_STATION_DIST, [0.0, 32.29, 1.0], units='unitless'
        )
        aviary_options.set_val(Aircraft.Wing.LOAD_DISTRIBUTION_CONTROL, 2.0, units='unitless')
        aviary_options.set_val(Aircraft.Wing.NUM_INTEGRATION_STATIONS, 50, units='unitless')
        aviary_options.set_val(Aircraft.Engine.NUM_FUSELAGE_ENGINES, 3, units='unitless')

        prob = self.prob = om.Problem()
        self.prob.model.add_subsystem(
            'wing_group',
            WingMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['aircraft:*'],
        )

        # EnginePodMass
        prob.model.set_input_defaults(Mission.Design.GROSS_MASS, 874099, units='lbm')
        prob.model.set_input_defaults(Aircraft.Electrical.MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Fuel.FUEL_SYSTEM_MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Hydraulics.MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Instruments.MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Nacelle.MASS, [0], units='lbm')
        prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, 0, units='lbm'
        )
        prob.model.set_input_defaults(Aircraft.Engine.MASS, [0], units='lbm')
        prob.model.set_input_defaults(Aircraft.Propulsion.TOTAL_STARTER_MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.THRUST_REVERSERS_MASS, 0, units='lbm')
        prob.model.set_input_defaults(Aircraft.Engine.SCALED_SLS_THRUST, [70000], units='lbf')
        prob.model.set_input_defaults(
            Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, 1900000.0, units='lbf'
        )
        # BWBDetailedWingBendingFact
        prob.model.set_input_defaults(
            'BWB_LOAD_PATH_SWEEP_DIST', [0.0, 15.337244816, 15.337244816], units='deg'
        )
        prob.model.set_input_defaults(
            'BWB_THICKNESS_TO_CHORD_DIST', [0.11, 0.11, 0.11], units='unitless'
        )
        prob.model.set_input_defaults(
            'BWB_CHORD_PER_SEMISPAN_DIST', [137.5, 91.3717, 14.2848], units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO, 3.4488821, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.ASPECT_RATIO_REF, 3.4488821, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.STRUT_BRACING_FACTOR, 0.0, units='unitless')
        prob.model.set_input_defaults(
            Aircraft.Wing.AEROELASTIC_TAILORING_FACTOR, 0.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD, 0.11, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.THICKNESS_TO_CHORD_REF, 0.11, units='unitless')
        # WingMiscMass
        prob.model.set_input_defaults(Aircraft.Wing.COMPOSITE_FRACTION, 1.0, units='unitless')
        # prob.model.set_input_defaults(Aircraft.Wing.AREA, 16555.972297926455, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Wing.MISC_MASS_SCALER, 1.0, units='unitless')
        # WingShearControlMass
        prob.model.set_input_defaults(
            Aircraft.Wing.CONTROL_SURFACE_AREA, 5513.13877521, units='ft**2'
        )
        prob.model.set_input_defaults(
            Aircraft.Wing.SHEAR_CONTROL_MASS_SCALER, 1.0, units='unitless'
        )
        # WingBendingMass
        prob.model.set_input_defaults(
            Aircraft.Wing.BENDING_MATERIAL_MASS_SCALER, 1.0, units='unitless'
        )
        prob.model.set_input_defaults(Aircraft.Wing.ENG_POD_INERTIA_FACTOR, 1.0, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.LOAD_FRACTION, 0.5311, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.SPAN, 238.080049, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.SWEEP, 35.7, units='deg')
        prob.model.set_input_defaults(Aircraft.Wing.ULTIMATE_LOAD_FACTOR, 3.75, units='unitless')
        prob.model.set_input_defaults(Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, 0.0, units='unitless')
        # WingBendingMass
        # BWBAftBodyMass
        prob.model.set_input_defaults(Aircraft.Fuselage.PLANFORM_AREA, val=7390.267, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.CABIN_AREA, val=5173.187, units='ft**2')
        prob.model.set_input_defaults(Aircraft.Fuselage.LENGTH, val=137.5, units='ft')
        prob.model.set_input_defaults(Aircraft.Wing.ROOT_CHORD, val=63.96, units='ft')
        prob.model.set_input_defaults('Rear_spar_percent_chord', 0.7, units='unitless')
        prob.model.set_input_defaults('Rear_spar_percent_chord_centerline', 0.7, units='unitless')

        setup_model_options(self.prob, aviary_options)
        prob.setup(check=False, force_alloc_complex=True)

    def test_case1(self):
        prob = self.prob
        prob.run_model()
        tol = 1e-9
        assert_near_equal(prob[Aircraft.Engine.POD_MASS], 0.0, tol)
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_FACTOR], 2.68745091, tol)
        assert_near_equal(prob[Aircraft.Wing.MISC_MASS], 21498.82990355, tol)
        assert_near_equal(prob[Aircraft.Wing.SHEAR_CONTROL_MASS], 38779.21499739, tol)
        assert_near_equal(prob[Aircraft.Wing.BENDING_MATERIAL_MASS], 6313.4476585, tol)
        assert_near_equal(prob[Aircraft.Fuselage.AFTBODY_MASS], 24278.05868511, tol)
        assert_near_equal(prob[Aircraft.Wing.BWB_AFTBODY_MASS], 20150.78870864, tol)
        assert_near_equal(prob[Aircraft.Wing.MASS], 86742.28126808, tol)


if __name__ == '__main__':
    # unittest.main()
    test = BWBWingGroupTest()
    test.setUp()
    test.test_case1()
