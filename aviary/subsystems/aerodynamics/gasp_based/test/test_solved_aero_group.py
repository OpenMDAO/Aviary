"""
This test validates passing a drag polar into the solved aero in the mission.
Computed lift and drag should be the same as reading the same polar in from
a file.
"""

import unittest
from copy import deepcopy

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_near_equal

from aviary.models.missions.height_energy_default import phase_info
from aviary.interface.methods_for_level2 import AviaryProblem
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.utils.csv_data_file import read_data_file
from aviary.utils.named_values import NamedValues
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft

# The drag-polar-generating component reads this in, instead of computing the polars.
polar_file = 'models/large_single_aisle_1/large_single_aisle_1_aero_free_reduced_alpha.csv'

phase_info = deepcopy(phase_info)

phase_info['pre_mission']['include_takeoff'] = False
phase_info['post_mission']['include_landing'] = False
phase_info['cruise']['subsystem_options']['core_aerodynamics']['method'] = 'tabular_cruise'
phase_info['cruise']['subsystem_options']['core_aerodynamics']['solve_alpha'] = True
phase_info['cruise']['subsystem_options']['core_aerodynamics']['aero_data'] = polar_file
phase_info.pop('climb')
phase_info.pop('descent')

data, _, _ = read_data_file(polar_file)
ALTITUDE = data.get_val('Altitude', 'ft')
MACH = data.get_val('Mach', 'unitless')
ALPHA = data.get_val('Angle_of_Attack', 'deg')

shape = (np.unique(ALTITUDE).size, np.unique(MACH).size, np.unique(ALPHA).size)
CL = data.get_val('CL').reshape(shape)
CD = data.get_val('CD').reshape(shape)


class TestSolvedAero(unittest.TestCase):
    def get_baseline_tabular_results(self):
        # Get the CL, CD of the baseline tabular aero problem
        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            local_phase_info,
        )
        prob.model.aero_method = LegacyCode.GASP

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        CL_base = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CL')
        CD_base = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CD')

        return CL_base, CD_base

    def test_solved_aero_pass_polar(self):
        # Test that passing training data provides the same results
        # Lift and Drag polars passed from external component in pre-mission.
        CL_base, CD_base = self.get_baseline_tabular_results()

        ph_in = deepcopy(phase_info)

        polar_builder = FakeDragPolarBuilder(name='aero', altitude=ALTITUDE, mach=MACH, alpha=ALPHA)
        aero_data = NamedValues()
        aero_data.set_val('altitude', ALTITUDE, 'ft')
        aero_data.set_val('mach', MACH, 'unitless')
        aero_data.set_val('angle_of_attack', ALPHA, 'deg')

        subsystem_options = {
            'method': 'tabular_cruise',
            'solve_alpha': True,
            'aero_data': aero_data,
            'connect_training_data': True,
        }
        ph_in['pre_mission']['external_subsystems'] = [polar_builder]

        ph_in['cruise']['subsystem_options'] = {'core_aerodynamics': subsystem_options}

        prob = AviaryProblem()

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            ph_in,
        )
        prob.model.aero_method = LegacyCode.GASP

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.aviary_inputs.set_val(Aircraft.Design.LIFT_POLAR, np.zeros_like(CL), units='unitless')
        prob.aviary_inputs.set_val(Aircraft.Design.DRAG_POLAR, np.zeros_like(CD), units='unitless')

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        CL_pass = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CL')
        CD_pass = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CD')

        assert_near_equal(CL_pass, CL_base, 1e-6)
        assert_near_equal(CD_pass, CD_base, 1e-6)

        # Test the drag scaler

        prob.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 2.0, units='unitless')

        prob.run_model()

        CL_pass = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CL')
        CD_pass = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CD')

        assert_near_equal(CL_pass, CL_base, 1e-6)
        assert_near_equal(CD_pass, 2.0 * CD_base, 1e-6)

    def test_parameters(self):
        # This test is to make sure that the aero builder creates a parameter
        # for wing area. It addresses a bug where this was absent.

        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()

        prob.load_inputs(
            'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv',
            local_phase_info,
        )
        prob.model.aero_method = LegacyCode.GASP

        # Change value just to be certain.
        prob.aviary_inputs.set_val(Aircraft.Wing.AREA, 7777, units='ft**2')

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        # verify that we are promoting the parameters.
        wing_area = prob.get_val('traj.cruise.rhs_all.aircraft:wing:area', units='ft**2')
        actual_wing_area = prob.aviary_inputs.get_val(Aircraft.Wing.AREA, units='ft**2')
        assert_near_equal(wing_area, actual_wing_area)

    def test_solved_aero_pass_polar_unique_abscissa(self):
        # Solved Aero with shortened lists of table abscissa.
        local_phase_info = deepcopy(phase_info)

        prob = AviaryProblem()

        csv_path = 'subsystems/aerodynamics/flops_based/test/data/high_wing_single_aisle.csv'
        prob.load_inputs(csv_path, local_phase_info)
        prob.model.aero_method = LegacyCode.GASP

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        CL_base = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CL')
        CD_base = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CD')

        # Lift and Drag polars passed from external component in pre-mission.

        ph_in = deepcopy(phase_info)

        alt = np.array(
            [
                0.0,
                3000.0,
                6000.0,
                9000.0,
                12000.0,
                15000.0,
                18000.0,
                21000.0,
                24000.0,
                27000.0,
                30000.0,
                33000.0,
                36000.0,
                38000.0,
                42000.0,
            ]
        )
        mach = np.array([0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9])
        alpha = np.array([-2.0, 0.0, 2.0, 4.0, 6.0, 8.0, 10.0])

        polar_builder = FakeDragPolarBuilder(name='aero', altitude=alt, mach=mach, alpha=alpha)
        aero_data = NamedValues()
        aero_data.set_val('altitude', alt, 'ft')
        aero_data.set_val('mach', mach, 'unitless')
        aero_data.set_val('angle_of_attack', alpha, 'deg')

        subsystem_options = {
            'method': 'tabular_cruise',
            'solve_alpha': True,
            'aero_data': aero_data,
            'connect_training_data': True,
        }
        ph_in['pre_mission']['external_subsystems'] = [polar_builder]

        ph_in['cruise']['subsystem_options'] = {'core_aerodynamics': subsystem_options}

        prob = AviaryProblem()

        prob.load_inputs(csv_path, ph_in)
        prob.model.aero_method = LegacyCode.GASP

        # Preprocess inputs
        prob.check_and_preprocess_inputs()

        prob.aviary_inputs.set_val(Aircraft.Design.LIFT_POLAR, np.zeros_like(CL), units='unitless')
        prob.aviary_inputs.set_val(Aircraft.Design.DRAG_POLAR, np.zeros_like(CD), units='unitless')

        prob.add_pre_mission_systems()
        prob.add_phases()
        prob.add_post_mission_systems()

        prob.link_phases()

        prob.setup()

        prob.set_initial_guesses()

        prob.run_model()

        CL_pass = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CL')
        CD_pass = prob.get_val('traj.cruise.rhs_all.core_aerodynamics.CD')

        assert_near_equal(CL_pass, CL_base, 1e-6)
        assert_near_equal(CD_pass, CD_base, 1e-6)


class FakeCalcDragPolar(om.ExplicitComponent):
    """
    This component is a stand-in for an externally computed lift/drag table
    calculation. It does nothing but read in the pre-computed table.
    """

    def initialize(self):
        """Declare options."""
        self.options.declare(
            'altitude',
            default=None,
            allow_none=True,
            desc='List of altitudes in ascending order.',
        )
        self.options.declare(
            'mach',
            default=None,
            allow_none=True,
            desc='List of mach numbers in ascending order.',
        )
        self.options.declare(
            'alpha',
            default=None,
            allow_none=True,
            desc='List of angles of attack in ascending order.',
        )

    def setup(self):
        altitude = self.options['altitude']
        mach = self.options['mach']
        alpha = self.options['alpha']

        self.add_input(Aircraft.Wing.AREA, 1.0, units='ft**2')
        self.add_input(Aircraft.Wing.SPAN, 1.0, units='ft')

        shape = (len(altitude), len(mach), len(alpha))

        self.add_output('drag_table', shape=shape, units='unitless')
        self.add_output('lift_table', shape=shape, units='unitless')

    def compute(self, inputs, outputs):
        """
        This component doesn't do anything, except set the drag and lift
        polars from the file we read in.

        Any real analysis would compute these tables.
        """
        outputs['drag_table'] = CD
        outputs['lift_table'] = CL


class FakeDragPolarBuilder(SubsystemBuilderBase):
    """
    Prototype of a subsystem that overrides an aviary internally computed var.

    Parameters
    ----------
    altitude : list or None
        List of altitudes in ascending order. (Optional)
    mach : list or None
        List of Mach numbers in ascending order. (Optional)
    alpha : list or None
        List of angles of attack in ascending order. (Optional)
    """

    def __init__(self, name='aero', altitude=None, mach=None, alpha=None):
        super().__init__(name)
        self.altitude = np.unique(altitude)
        self.mach = np.unique(mach)
        self.alpha = np.unique(alpha)

    def build_pre_mission(self, aviary_inputs):
        """
        Build an OpenMDAO system for the pre-mission computations of the subsystem.

        Returns
        -------
        pre_mission_sys : openmdao.core.Group
            An OpenMDAO group containing all computations that need to happen in
            the pre-mission part of the Aviary problem. This
            includes sizing, design, and other non-mission parameters.
        """
        group = om.Group()

        calc_drag_polar = FakeCalcDragPolar(
            altitude=self.altitude, mach=self.mach, alpha=self.alpha
        )

        group.add_subsystem(
            'aero',
            calc_drag_polar,
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[
                ('drag_table', Aircraft.Design.DRAG_POLAR),
                ('lift_table', Aircraft.Design.LIFT_POLAR),
            ],
        )
        return group


if __name__ == '__main__':
    unittest.main()
    # test = TestSolvedAero()
    # test.test_solved_aero_pass_polar_unique_abscissa()
