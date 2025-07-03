import unittest
from copy import deepcopy

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal
from openmdao.utils.testing_utils import use_tempdirs
from parameterized import parameterized

from aviary.interface.default_phase_info.height_energy import phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.aerodynamics.aerodynamics_builder import CoreAerodynamicsBuilder
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.premission import CorePreMission
from aviary.subsystems.propulsion.utils import build_engine_deck
from aviary.utils.aviary_values import AviaryValues
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.named_values import NamedValues
from aviary.utils.test_utils.default_subsystems import get_default_premission_subsystems
from aviary.validation_cases.validation_tests import get_flops_inputs, get_flops_outputs, print_case
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission, Settings

FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP

CDI_table = 'subsystems/aerodynamics/flops_based/test/large_single_aisle_1_CDI_polar.csv'
CD0_table = 'subsystems/aerodynamics/flops_based/test/large_single_aisle_1_CD0_polar.csv'


@use_tempdirs
class TabularAeroGroupFileTest(unittest.TestCase):
    # Test drag comp with data from file, structured grid
    def setUp(self):
        self.prob = om.Problem()
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 0)

        kwargs = {'method': 'tabular', 'CDI_data': CDI_table, 'CD0_data': CD0_table}

        aero_builder = CoreAerodynamicsBuilder(code_origin=FLOPS)

        self.prob.model.add_subsystem(
            'aero',
            aero_builder.build_mission(num_nodes=1, aviary_inputs=aviary_options, **kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(self.prob, aviary_options)

        self.prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, val=0.3876, units='unitless')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        # TODO currently no way to use FLOPS test case data for mission components
        # test data from large_single_aisle_2 climb profile
        # tabular aero was set to large_single_aisle_1, expected value adjusted accordingly
        self.prob.set_val(
            Dynamic.Mission.VELOCITY, val=115, units='m/s'
        )  # convert from knots to ft/s
        self.prob.set_val(Dynamic.Mission.ALTITUDE, val=10582, units='m')
        self.prob.set_val(Dynamic.Vehicle.MASS, val=80442, units='kg')
        # 1344.5? 'reference' vs 'calculated'?
        self.prob.set_val(Aircraft.Wing.AREA, val=1341, units='ft**2')
        # calculated from online atmospheric table
        self.prob.set_val(Dynamic.Atmosphere.DENSITY, val=0.88821, units='kg/m**3')

        self.prob.run_model()

        # We know that computed drag (from FLOPS) is higher than what this tabular data
        # computes. Use loose tolerance.
        tol = 0.03

        assert_near_equal(
            self.prob.get_val(Dynamic.Vehicle.DRAG, units='N'), 53934.78861492, tol
        )  # check the value of each output

        # TODO resolve partials wrt gravity (decide on implementation of gravity)
        partial_data = self.prob.check_partials(out_stream=None, method='cs', step=1.1e-40)
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-12)  # check the partial derivatives

    def test_parameters(self):
        local_phase_info = deepcopy(phase_info)
        core_aero = local_phase_info['cruise']['subsystem_options']['core_aerodynamics']

        core_aero['method'] = 'tabular'
        core_aero['CDI_data'] = CDI_table
        core_aero['CD0_data'] = CD0_table
        local_phase_info.pop('climb')
        local_phase_info.pop('descent')

        prob = AviaryProblem(verbosity=0)

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            local_phase_info,
        )

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        print('about to run')
        prob.run_model()

        # verify that we are promoting the parameters.
        wing_area = prob.get_val('traj.cruise.rhs_all.aircraft:wing:area', units='ft**2')
        actual_wing_area = prob.aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2')
        assert_near_equal(wing_area, actual_wing_area)


@use_tempdirs
class TabularAeroGroupDataTest(unittest.TestCase):
    # Test tabular drag comp with training data, structured grid
    def setUp(self):
        self.prob = om.Problem()
        aviary_options = AviaryValues()
        aviary_options.set_val(Settings.VERBOSITY, 0)

        CDI_table = _default_CDI_data()
        CDI_values = CDI_table.get_val('lift_dependent_drag_coefficient')
        CD0_table = _default_CD0_data()
        CD0_values = CD0_table.get_val('zero_lift_drag_coefficient')

        drag_data = om.ExecComp()
        drag_data.add_output(
            Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR, CD0_values, units='unitless'
        )
        drag_data.add_output(
            Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR, CDI_values, units='unitless'
        )

        self.prob.model.add_subsystem('drag_data', drag_data, promotes_outputs=['*'])

        # Pass zero-arrays for the training data, so that this test won't pass unless
        # we are passing the values from the drag_data component.
        shape = CDI_table.get_item('lift_dependent_drag_coefficient')[0].shape
        CDI_clean_table = _default_CDI_data()
        CDI_clean_table.set_val('lift_dependent_drag_coefficient', np.zeros(shape))

        shape = CD0_table.get_item('zero_lift_drag_coefficient')[0].shape
        CD0_clean_table = _default_CD0_data()
        CD0_clean_table.set_val('zero_lift_drag_coefficient', np.zeros(shape))

        self.CDI_values = CDI_values
        self.CD0_values = CD0_values
        self.CDI_clean_table = CDI_clean_table
        self.CD0_clean_table = CD0_clean_table

        kwargs = {
            'method': 'tabular',
            'CDI_data': CDI_clean_table,
            'CD0_data': CD0_clean_table,
            'connect_training_data': True,
        }

        aero_builder = CoreAerodynamicsBuilder(code_origin=FLOPS)

        self.prob.model.add_subsystem(
            'aero',
            aero_builder.build_mission(num_nodes=1, aviary_inputs=aviary_options, **kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(self.prob, aviary_options)

        self.prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, val=0.3876, units='unitless')

        self.prob.setup(check=False, force_alloc_complex=True)

    def test_case(self):
        # TODO currently no way to use FLOPS test case data for mission components
        # test data from large_single_aisle_2 climb profile
        # tabular aero was set to large_single_aisle_1 data, expected value adjusted accordingly
        self.prob.set_val(
            Dynamic.Mission.VELOCITY, val=115, units='m/s'
        )  # convert from knots to ft/s
        self.prob.set_val(Dynamic.Mission.ALTITUDE, val=10582, units='m')
        self.prob.set_val(Dynamic.Vehicle.MASS, val=80442, units='kg')
        # 1344.5? 'reference' vs 'calculated'?
        self.prob.set_val(Aircraft.Wing.AREA, val=1341, units='ft**2')
        # calculated from online atmospheric table
        self.prob.set_val(Dynamic.Atmosphere.DENSITY, val=0.88821, units='kg/m**3')

        self.prob.run_model()

        # We know that computed drag (from FLOPS) is higher than what this tabular data
        # computes. Use loose tolerance.
        tol = 0.03

        assert_near_equal(
            self.prob.get_val(Dynamic.Vehicle.DRAG, units='N'), 53934.78861492, tol
        )  # check the value of each output

        # TODO resolve partials wrt gravity (decide on implementation of gravity)
        partial_data = self.prob.check_partials(out_stream=None, method='cs', step=1.1e-40)
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-12)  # check the partial derivatives

    def test_parameters(self):
        local_phase_info = deepcopy(phase_info)
        core_aero = local_phase_info['cruise']['subsystem_options']['core_aerodynamics']

        core_aero['method'] = 'tabular'
        core_aero['connect_training_data'] = True
        core_aero['CDI_data'] = self.CDI_clean_table
        core_aero['CD0_data'] = self.CD0_clean_table
        local_phase_info.pop('climb')
        local_phase_info.pop('descent')

        # This is a somewhat contrived problem, so pick a mass.
        local_phase_info['cruise']['user_options']['mass_initial'] = (150000.0, 'lbm')

        prob = AviaryProblem()

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            local_phase_info,
        )

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        # Connect or set.
        prob.aviary_inputs.set_val(Aircraft.Design.LIFT_INDEPENDENT_DRAG_POLAR, self.CD0_values)
        prob.aviary_inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_POLAR, self.CDI_values)

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        assert_near_equal(prob.get_val('traj.cruise.rhs_all.drag', units='lbf')[0], 9907.0, 1.0e-3)


data_sets = ['LargeSingleAisle1FLOPS', 'LargeSingleAisle2FLOPS', 'N3CC']


class ComputedVsTabularTest(unittest.TestCase):
    @parameterized.expand(data_sets, name_func=print_case)
    def test_case(self, case_name):
        flops_inputs = get_flops_inputs(case_name, preprocess=True)
        flops_outputs = get_flops_outputs(case_name)

        flops_inputs.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 0.9)
        flops_inputs.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.1)
        flops_inputs.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 0.9)
        flops_inputs.set_val(Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR, 1.1)

        key = Aircraft.Propulsion.TOTAL_NUM_ENGINES
        flops_inputs.set_val(key, flops_outputs.get_val(key))

        key = Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST
        flops_inputs.set_val(key, *(flops_outputs.get_item(key)))

        flops_inputs.set_val(Settings.VERBOSITY, 0)

        mass, units = flops_inputs.get_item(Mission.Design.GROSS_MASS)
        mass = mass * 0.92

        alt = 10582
        alt_units = 'm'

        vel = 115
        vel_units = 'm/s'

        dynamic_inputs = AviaryValues()

        dynamic_inputs.set_val(Dynamic.Mission.VELOCITY, val=vel, units=vel_units)
        dynamic_inputs.set_val(Dynamic.Mission.ALTITUDE, val=alt, units=alt_units)
        dynamic_inputs.set_val(Dynamic.Vehicle.MASS, val=mass, units=units)

        prob = _get_computed_aero_data_at_altitude(alt, alt_units)

        sos = prob.get_val(Dynamic.Atmosphere.SPEED_OF_SOUND, vel_units)
        mach = vel / sos

        dynamic_inputs.set_val(Dynamic.Atmosphere.MACH, val=mach, units='unitless')

        key = Dynamic.Atmosphere.DENSITY
        units = 'kg/m**3'
        val = prob.get_val(key, units)

        dynamic_inputs.set_val(key, val=val, units=units)

        key = Dynamic.Atmosphere.TEMPERATURE
        units = 'degR'
        val = prob.get_val(key, units)

        dynamic_inputs.set_val(key, val=val, units=units)

        key = Dynamic.Atmosphere.STATIC_PRESSURE
        units = 'N/m**2'
        val = prob.get_val(key, units)

        dynamic_inputs.set_val(key, val=val, units=units)

        prob = _run_computed_aero_harness(flops_inputs, dynamic_inputs, 1)

        computed_drag = prob.get_val(Dynamic.Vehicle.DRAG, 'N')

        CDI_data, CD0_data = _computed_aero_drag_data(
            flops_inputs, *_design_altitudes.get_item(case_name)
        )

        prob = om.Problem()

        kwargs = {'method': 'tabular', 'CDI_data': CDI_data, 'CD0_data': CD0_data}

        aero_builder = CoreAerodynamicsBuilder(code_origin=FLOPS)

        prob.model.add_subsystem(
            'aero',
            aero_builder.build_mission(num_nodes=1, aviary_inputs=flops_inputs, **kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        setup_model_options(prob, flops_inputs)

        prob.model.set_input_defaults(Dynamic.Atmosphere.MACH, val=0.3876, units='unitless')

        prob.setup(check=False, force_alloc_complex=True)

        for key, (val, units) in dynamic_inputs:
            try:
                prob.set_val(key, val, units)
            except:
                pass  # unused variable

        set_aviary_initial_values(prob, flops_inputs)

        prob.run_model()

        tabular_drag = prob.get_val(Dynamic.Vehicle.DRAG, 'N')

        assert_near_equal(tabular_drag, computed_drag, 0.005)

        partial_data = prob.check_partials(out_stream=None, method='cs', step=1.1e-40)
        assert_check_partials(partial_data, atol=1e-9, rtol=1e-12)


# region - hardcoded data from large_single_aisle_1 (fixed alt cruise variant) FLOPS model
def _default_CD0_data():
    alt_range = _make_default_alt_range()  # len == 13
    mach_range = _make_default_mach_range()  # len == 12

    # fmt: off
    CD0 = np.array([
        [
            0.02182, 0.02228, 0.02278, 0.02332, 0.02389, 0.02451,
            0.02518, 0.0259, 0.02689, 0.02801, 0.02921, 0.03049,
        ],
        [
            0.03184, 0.02043, 0.02086, 0.02132, 0.0218, 0.02233,
            0.02289, 0.02349, 0.02414, 0.02504, 0.02605, 0.02713,
        ],
        [
            0.02827, 0.02949, 0.01946, 0.01987, 0.02029, 0.02075,
            0.02123, 0.02176, 0.02232, 0.02293, 0.02377, 0.02471,
        ],
        [
            0.0257, 0.02676, 0.02788, 0.0187, 0.01908, 0.01949,
            0.01992, 0.02038, 0.02088, 0.02141, 0.02199, 0.02278,
        ],
        [
            0.02366, 0.0246, 0.0256, 0.02665, 0.01824, 0.0186,
            0.01899, 0.0194, 0.01984, 0.02031, 0.02082, 0.02137,
        ],
        [
            0.02212, 0.02297, 0.02386, 0.02481, 0.02581, 0.01828,
            0.01863, 0.019, 0.0194, 0.01982, 0.02027, 0.02076,
        ],
        [
            0.02129, 0.02201, 0.02282, 0.02368, 0.02458, 0.02554,
            0.01843, 0.01878, 0.01914, 0.01953, 0.01994, 0.02039,
        ],
        [
            0.02087, 0.02139, 0.02209, 0.02289, 0.02373, 0.02461,
            0.02555, 0.01864, 0.01898, 0.01934, 0.01972, 0.02014,
        ],
        [
            0.02058, 0.02105, 0.02157, 0.02227, 0.02305, 0.02388,
            0.02476, 0.02569, 0.01909, 0.01943, 0.01979, 0.02017,
        ],
        [
            0.02058, 0.02101, 0.02149, 0.02199, 0.02269, 0.02347,
            0.02429, 0.02516, 0.02608, 0.02021, 0.02054, 0.02089,
        ],
        [
            0.02127, 0.02168, 0.02211, 0.02258, 0.02308, 0.02377,
            0.02454, 0.02535, 0.02622, 0.02713, 0.02222, 0.02255,
        ],
        [
            0.0229, 0.02328, 0.02368, 0.02411, 0.02457, 0.02507,
            0.02575, 0.02651, 0.02732, 0.02818, 0.02908, 0.03704,
        ],
        [
            0.03737, 0.03772, 0.03809, 0.03849, 0.03891, 0.03937,
            0.03987, 0.04054, 0.0413, 0.0421, 0.04294, 0.04384,
        ],
    ])
    # fmt: on

    # mach_list, alt_list = np.meshgrid(mach_range, alt_range)

    CD0 = np.array(CD0)  # .flatten()
    # mach_list = np.array(mach_list).flatten()
    # alt_list = np.array(alt_list).flatten()

    CD0_data = NamedValues()
    CD0_data.set_val(Dynamic.Mission.ALTITUDE, alt_range, 'ft')
    CD0_data.set_val(Dynamic.Atmosphere.MACH, mach_range)
    CD0_data.set_val('zero_lift_drag_coefficient', CD0)

    return CD0_data


def _default_CDI_data():
    mach_range = _make_default_mach_range()  # len == 12
    cl_range = _make_default_cl_range()  # len == 15

    # fmt: off
    CDI = np.array([
        [
            0.00114, 0.00138, 0.0019, 0.0027, 0.00378, 0.00511, 0.00669,
            0.00837, 0.0104, 0.01295, 0.01558, 0.01837, 0.0215, 0.025, 0.02888,
        ],  # 0
        [
            0.00113, 0.00137, 0.0019, 0.0027, 0.00377, 0.00511, 0.00669,
            0.00837, 0.0104, 0.01295, 0.01558, 0.01837, 0.0215, 0.025, 0.02888,
        ],  # 1
        [
            0.00114, 0.00138, 0.0019, 0.0027, 0.00378, 0.00511, 0.00669,
            0.00837, 0.0104, 0.01295, 0.01558, 0.01837, 0.0215, 0.025, 0.02888,
        ],  # 2
        [
            0.00117, 0.00138, 0.0019, 0.0027, 0.00379, 0.00513, 0.0067,
            0.00837, 0.01041, 0.01295, 0.01558, 0.01837, 0.0215, 0.025, 0.02888,
        ],  # 3
        [
            0.00121, 0.0014, 0.0019, 0.0027, 0.00381, 0.00515, 0.0067,
            0.00838, 0.01042, 0.01294, 0.01558, 0.01837, 0.0215, 0.025, 0.02888,
        ],  # 4
        [
            0.00132, 0.00144, 0.0019, 0.00271, 0.00387, 0.00522, 0.00673,
            0.00839, 0.01043, 0.01294, 0.01558, 0.01837, 0.0215, 0.025, 0.02888,
        ],  # 5
        [
            0.00142, 0.00148, 0.00192, 0.00273, 0.00392, 0.00529, 0.00675,
            0.00841, 0.01045, 0.01293, 0.01556, 0.01841, 0.02169, 0.02542, 0.02961,
        ],  # 6
        [
            0.0015, 0.00152, 0.00194, 0.00276, 0.00397, 0.00534, 0.00678,
            0.00842, 0.01046, 0.0129, 0.01562, 0.01891, 0.02291, 0.02769, 0.03327,
        ],  # 7
        [
            0.00166, 0.00159, 0.00196, 0.00279, 0.00406, 0.00549, 0.00693,
            0.00855, 0.01065, 0.01328, 0.01686, 0.02156, 0.02667, 0.03227, 0.03832,
        ],  # 8
        [
            0.00208, 0.00175, 0.00201, 0.00285, 0.00428, 0.00592, 0.0076,
            0.00965, 0.01245, 0.01657, 0.0215, 0.02697, 0.0341, 0.04264, 0.0526,
        ],  # 9
        [
            0.00336, 0.00221, 0.00209, 0.003, 0.00494, 0.00757, 0.01074,
            0.01452, 0.01987, 0.02654, 0.03295, 0.03957, 0.04634, 0.05334, 0.06059,
        ],  # 10
        [
            0.00636, 0.00327, 0.00224, 0.00324, 0.00627, 0.01154, 0.01915,
            0.02577, 0.03848, 0.05251, 0.05737, 0.06445, 0.0703, 0.07582, 0.08127,
        ],
    ])  # 11
    # fmt: on

    # cl_list, mach_list = np.meshgrid(cl_range, mach_range)

    CDI = np.array(CDI)  # .flatten()
    # cl_list = np.array(cl_list).flatten()
    # mach_list = np.array(mach_list).flatten()
    CDI_data = NamedValues()
    CDI_data.set_val(Dynamic.Atmosphere.MACH, mach_range)
    CDI_data.set_val('lift_coefficient', cl_range)
    CDI_data.set_val('lift_dependent_drag_coefficient', CDI)

    return CDI_data


# fmt: off
def _make_default_alt_range():
    alt_range = np.array([
        0, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000
    ])

    return alt_range


def _make_default_mach_range():
    mach_range = np.array([
        0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.750, 0.775, 0.800, 0.825, 0.850, 0.875
    ])

    return mach_range


def _make_default_cl_range():
    cl_range = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85])

    return cl_range
# fmt: on

# endregion - hardcoded data from large_single_aisle_1 (fixed alt cruise variant) FLOPS model


# region - computed aero data
def _computed_aero_drag_data(flops_inputs: AviaryValues, design_altitude, units):
    nsteps = 12

    start = 0.0
    stop = 70000.0
    step = (stop - start) / nsteps
    alt = np.arange(start, stop, step)  # ft

    start = 0.2
    stop = 1.0
    step = (stop - start) / nsteps
    seed = np.arange(start, stop, step)

    mach = np.repeat(seed, nsteps)  # unitless
    CL = np.tile(seed, nsteps)  # unitless

    S = flops_inputs.get_val(Aircraft.Wing.AREA, 'ft**2')

    # calculate temperature (degR), static pressure (lbf/ft**2), and mass (lbm) at design
    # altitude from lift coefficients and Mach numbers
    prob: om.Problem = _get_computed_aero_data_at_altitude(design_altitude, units)
    T = prob.get_val(Dynamic.Atmosphere.TEMPERATURE, 'degR')
    P = prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, 'lbf/ft**2')

    mass = CL * S * 0.5 * 1.4 * P * mach**2  # lbf -> lbm * 1g

    # calculate lift-dependent drag coefficient table, including pressure drag effects
    nn = len(mass)

    dynamic_inputs = AviaryValues()

    dynamic_inputs.set_val(Dynamic.Atmosphere.MACH, val=mach)
    dynamic_inputs.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P, units='lbf/ft**2')
    dynamic_inputs.set_val(Dynamic.Atmosphere.TEMPERATURE, val=T, units='degR')
    dynamic_inputs.set_val(Dynamic.Vehicle.MASS, val=mass, units='lbm')

    prob = _run_computed_aero_harness(flops_inputs, dynamic_inputs, nn)

    CDI: np.ndarray = prob.get_val('CDI')
    CDI = np.reshape(CDI.flatten(), (nsteps, nsteps))

    CDI_data = NamedValues()
    CDI_data.set_val(Dynamic.Atmosphere.MACH, seed)
    CDI_data.set_val('lift_coefficient', seed)
    CDI_data.set_val('lift_dependent_drag_coefficient', CDI)

    # calculate lift-independent drag coefficient table
    mass, units = flops_inputs.get_item(Mission.Design.GROSS_MASS)
    mass = mass * 0.9

    mach = seed
    nn = len(mach)

    dynamic_inputs = AviaryValues()

    dynamic_inputs.set_val(Dynamic.Atmosphere.MACH, val=mach)
    dynamic_inputs.set_val(Dynamic.Vehicle.MASS, val=mass, units=units)

    CD0 = []

    for h in alt:
        prob: om.Problem = _get_computed_aero_data_at_altitude(h, 'ft')
        T = prob.get_val(Dynamic.Atmosphere.TEMPERATURE, 'degR')
        P = prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, 'lbf/ft**2')

        dynamic_inputs.set_val(Dynamic.Atmosphere.STATIC_PRESSURE, val=P, units='lbf/ft**2')
        dynamic_inputs.set_val(Dynamic.Atmosphere.TEMPERATURE, val=T, units='degR')

        prob = _run_computed_aero_harness(flops_inputs, dynamic_inputs, nn)

        CD0.append(prob.get_val('CD0'))

    CD0 = np.array(CD0)

    CD0_data = NamedValues()
    CD0_data.set_val(Dynamic.Mission.ALTITUDE, alt, 'ft')
    CD0_data.set_val(Dynamic.Atmosphere.MACH, seed)
    CD0_data.set_val('zero_lift_drag_coefficient', CD0)

    return (CDI_data, CD0_data)


def _get_computed_aero_data_at_altitude(altitude, units):
    prob = om.Problem()

    prob.model.add_subsystem(name='atmosphere', subsys=Atmosphere(num_nodes=1), promotes=['*'])

    prob.setup()

    prob.set_val(Dynamic.Mission.ALTITUDE, altitude, units)

    prob.run_model()

    return prob


def _run_computed_aero_harness(flops_inputs, dynamic_inputs, num_nodes):
    prob = om.Problem(_ComputedAeroHarness(num_nodes=num_nodes, aviary_options=flops_inputs))

    setup_model_options(prob, flops_inputs)

    prob.setup()

    set_aviary_initial_values(prob, dynamic_inputs)
    set_aviary_initial_values(prob, flops_inputs)

    prob.run_model()

    return prob


class _ComputedAeroHarness(om.Group):
    """Calculate drag and drag polars."""

    def initialize(self):
        options = self.options

        options.declare('num_nodes', types=int)

        options.declare('gamma', default=1.4, desc='Ratio of specific heats for air.')

        options.declare(
            'aviary_options',
            types=AviaryValues,
            desc='collection of Aircraft/Mission specific options',
        )

    def setup(self):
        options = self.options

        nn = options['num_nodes']
        gamma = options['gamma']
        aviary_options: AviaryValues = options['aviary_options']

        engines = [build_engine_deck(aviary_options)]
        # don't need mass, skip it
        default_premission_subsystems = get_default_premission_subsystems('FLOPS', engines)[:-1]

        # Upstream pre-mission analysis for aero
        pre_mission: om.Group = self.add_subsystem(
            'pre_mission',
            CorePreMission(aviary_options=aviary_options, subsystems=default_premission_subsystems),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )

        kwargs = {'method': 'computed', 'gamma': gamma}

        aero_builder = CoreAerodynamicsBuilder(code_origin=FLOPS)

        self.add_subsystem(
            'aero',
            aero_builder.build_mission(num_nodes=nn, aviary_inputs=aviary_options, **kwargs),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        # key = Aircraft.Engine.SCALED_SLS_THRUST
        key = Aircraft.Engine.SCALE_FACTOR
        val, units = aviary_options.get_item(key)
        pre_mission.set_input_defaults(key, val, units)


_design_altitudes = AviaryValues(
    {
        'LargeSingleAisle1FLOPS': (41000, 'ft'),
        'LargeSingleAisle2FLOPS': (41000, 'ft'),
        'N3CC': (43000, 'ft'),
    }
)
# endregion - computed aero data


if __name__ == '__main__':
    unittest.main()
    # test = TabularAeroGroupDataTest()
    # test.setUp()
    # test.test_parameters()
