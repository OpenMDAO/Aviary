import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import \
    distributed_engine_count_factor
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission

# TODO: update non-transport components to new standard to remove these variables
_wing_engine_count_factor = 'aircraft:propulsion:control:wing_engine_count_factor'
_fuse_engine_count_factor = 'aircraft:propulsion:control:fuselage_engine_count_factor'
_max_mach = 'aircraft:design:dimensions:max_mach'


class TransportHydraulicsGroupMass(om.ExplicitComponent):
    '''
    Calculates the mass of the hydraulics group using the transport/general aviation
    method. The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.

    ASSUMPTIONS: all engines use hydraulics which follow this equation
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, val=0.0)

        add_aviary_input(self, Aircraft.Hydraulics.SYSTEM_PRESSURE, val=0.0)

        add_aviary_input(self, Aircraft.Hydraulics.MASS_SCALER, val=1.0)

        add_aviary_input(self, Aircraft.Wing.AREA, val=0.0)

        add_aviary_input(self, Aircraft.Wing.VAR_SWEEP_MASS_PENALTY, val=0.0)

        add_aviary_output(self, Aircraft.Hydraulics.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_wing_eng = aviary_options.get_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)
        num_fuse_eng = aviary_options.get_val(
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)
        num_wing_eng_fact = distributed_engine_count_factor(num_wing_eng)
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)

        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        sys_press = inputs[Aircraft.Hydraulics.SYSTEM_PRESSURE]
        scaler = inputs[Aircraft.Hydraulics.MASS_SCALER]
        area = inputs[Aircraft.Wing.AREA]
        var_sweep = inputs[Aircraft.Wing.VAR_SWEEP_MASS_PENALTY]
        max_mach = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        outputs[Aircraft.Hydraulics.MASS] = (
            0.57 * (planform + 0.27 * area)
            * (1 + 0.03 * num_wing_eng_fact + 0.05 * num_fuse_eng_fact)
            * (3000 / sys_press)**0.35 * (1 + 0.04 * var_sweep) * max_mach**0.33
            * scaler / GRAV_ENGLISH_LBM)

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        num_wing_eng = aviary_options.get_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES)
        num_fuse_eng = aviary_options.get_val(
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)
        num_wing_eng_fact = distributed_engine_count_factor(num_wing_eng)
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)

        planform = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        sys_press = inputs[Aircraft.Hydraulics.SYSTEM_PRESSURE]
        scaler = inputs[Aircraft.Hydraulics.MASS_SCALER]
        area = inputs[Aircraft.Wing.AREA]
        var_sweep = inputs[Aircraft.Wing.VAR_SWEEP_MASS_PENALTY]
        max_mach = aviary_options.get_val(Mission.Constraints.MAX_MACH)

        term1 = (planform + 0.27 * area)
        term2 = (1.0 + 0.03 * num_wing_eng_fact + 0.05 * num_fuse_eng_fact)
        term3 = (3000.0 / sys_press)**0.35
        term4 = (1.0 + 0.04 * var_sweep)
        term5 = max_mach**0.33

        J[Aircraft.Hydraulics.MASS, Aircraft.Fuselage.PLANFORM_AREA] = (
            0.57 * term2 * term3 * term4 * term5 * scaler / GRAV_ENGLISH_LBM)

        J[Aircraft.Hydraulics.MASS, Aircraft.Wing.AREA] = (
            0.1539 * term2 * term3 * term4 * term5 * scaler / GRAV_ENGLISH_LBM)

        J[Aircraft.Hydraulics.MASS, Aircraft.Hydraulics.SYSTEM_PRESSURE] = (
            -3.2880267277063872637891549227565 * term1 * term2
            * sys_press**-1.35 * term4 * term5 * scaler / GRAV_ENGLISH_LBM)

        J[Aircraft.Hydraulics.MASS, Aircraft.Wing.VAR_SWEEP_MASS_PENALTY] = (
            0.0228 * term1 * term2 * term3 * term5 * scaler / GRAV_ENGLISH_LBM)

        J[Aircraft.Hydraulics.MASS, Aircraft.Hydraulics.MASS_SCALER] = (
            0.57 * term1 * term2 * term3 * term4 * term5 / GRAV_ENGLISH_LBM)


class AltHydraulicsGroupMass(om.ExplicitComponent):
    '''
    Calculates the mass of the hydraulics group using the alternate method.
    The methodology is based on the FLOPS weight equations, modified to
    output mass instead of weight.
    '''

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Wing.AREA, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.WETTED_AREA, val=0.0)

        add_aviary_input(self, Aircraft.HorizontalTail.THICKNESS_TO_CHORD, val=0.0)

        add_aviary_input(self, Aircraft.VerticalTail.AREA, val=0.0)

        add_aviary_input(self, Aircraft.Hydraulics.MASS_SCALER, val=1.0)

        add_aviary_output(self, Aircraft.Hydraulics.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        area = inputs[Aircraft.Wing.AREA]
        horiz_wetted_area = inputs[Aircraft.HorizontalTail.WETTED_AREA]
        horiz_thick_chord = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        vert_area = inputs[Aircraft.VerticalTail.AREA]
        scaler = inputs[Aircraft.Hydraulics.MASS_SCALER]

        outputs[Aircraft.Hydraulics.MASS] = (
            0.6053 * (area + 1.44 * (horiz_wetted_area /
                      (2.0 + 0.387 * horiz_thick_chord) + vert_area)) * scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        area = inputs[Aircraft.Wing.AREA]
        horiz_wetted_area = inputs[Aircraft.HorizontalTail.WETTED_AREA]
        horiz_thick_chord = inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        thick_chord_term = \
            2.0 + 0.387 * inputs[Aircraft.HorizontalTail.THICKNESS_TO_CHORD]
        vert_area = inputs[Aircraft.VerticalTail.AREA]
        scaler = inputs[Aircraft.Hydraulics.MASS_SCALER]

        J[Aircraft.Hydraulics.MASS, Aircraft.Wing.AREA] = (
            0.6053 * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Hydraulics.MASS, Aircraft.HorizontalTail.WETTED_AREA] = (
            0.871632 / thick_chord_term * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Hydraulics.MASS, Aircraft.HorizontalTail.THICKNESS_TO_CHORD] = (
            -0.337321584 * horiz_wetted_area /
            (thick_chord_term * thick_chord_term) * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Hydraulics.MASS, Aircraft.VerticalTail.AREA] = (
            0.871632 * scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Hydraulics.MASS, Aircraft.Hydraulics.MASS_SCALER] = (
            0.6053 * (area + 1.44 * (horiz_wetted_area /
                      (2.0 + 0.387 * horiz_thick_chord) + vert_area)) / GRAV_ENGLISH_LBM
        )
