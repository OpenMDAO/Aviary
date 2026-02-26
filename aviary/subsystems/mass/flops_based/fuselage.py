import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import distributed_engine_count_factor
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission, Settings


class TransportFuselageMass(om.ExplicitComponent):
    """
    Computes the mass of the fuselage for transports. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Fuselage.MILITARY_CARGO_FLOOR)
        add_aviary_option(self, Aircraft.Fuselage.NUM_FUSELAGES)
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.REF_DIAMETER, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Fuselage.MASS, '*')

    def compute(self, inputs, outputs):
        length = inputs[Aircraft.Fuselage.LENGTH]
        scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        avg_diameter = inputs[Aircraft.Fuselage.REF_DIAMETER]

        num_fuse = self.options[Aircraft.Fuselage.NUM_FUSELAGES]
        num_fuse_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES]

        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)
        military_cargo = self.options[Aircraft.Fuselage.MILITARY_CARGO_FLOOR]

        mil_factor = 1.38 if military_cargo else 1.0

        outputs[Aircraft.Fuselage.MASS] = (
            scaler
            * 1.35
            * (avg_diameter * length) ** 1.28
            * (1.0 + 0.05 * num_fuse_eng_fact)
            * mil_factor
            * num_fuse
            / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        length = inputs[Aircraft.Fuselage.LENGTH]
        scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        avg_diameter = inputs[Aircraft.Fuselage.REF_DIAMETER]
        num_fuse = self.options[Aircraft.Fuselage.NUM_FUSELAGES]
        num_fuse_eng = self.options[Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES]
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)
        military_cargo = self.options[Aircraft.Fuselage.MILITARY_CARGO_FLOOR]

        # avg_diameter = (max_height + max_width) / 2.
        avg_diameter_exp = avg_diameter**1.28
        length_exp = length**1.28
        height_width_exp = (avg_diameter) ** 0.28
        mil_factor = 1.38 if military_cargo else 1.0
        addtl_factor = (1.0 + 0.05 * num_fuse_eng_fact) * mil_factor * num_fuse

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_SCALER] = (
            1.35 * avg_diameter_exp * length_exp * addtl_factor / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.LENGTH] = (
            scaler * 1.728 * avg_diameter_exp * length**0.28 * addtl_factor / GRAV_ENGLISH_LBM
        )
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.REF_DIAMETER] = (
            scaler * 1.728 * length_exp * height_width_exp * addtl_factor / GRAV_ENGLISH_LBM
        )


class AltFuselageMass(om.ExplicitComponent):
    """
    Computes the mass of the fuselage using the alternate method. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER, units='unitless')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='ft')
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='ft')

        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        mass_scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        fuse_wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]
        fuse_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        fuse_width = inputs[Aircraft.Fuselage.MAX_WIDTH]

        outputs[Aircraft.Fuselage.MASS] = (
            3.939
            * fuse_wetted_area
            / (fuse_height / fuse_width) ** 0.221
            * mass_scaler
            / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        mass_scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        fuse_wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]
        fuse_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        fuse_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        fuse_height_fact = fuse_height**0.221
        fuse_width_fact = fuse_width**0.221
        total_fact = fuse_height_fact / fuse_width_fact

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_SCALER] = (
            3.939 * fuse_wetted_area / total_fact / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = (
            3.939 / total_fact * mass_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MAX_HEIGHT] = (
            -0.870519
            * fuse_wetted_area
            * fuse_width_fact
            * fuse_height**-1.221
            * mass_scaler
            / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MAX_WIDTH] = (
            0.870519
            * fuse_wetted_area
            * fuse_width**-0.779
            / fuse_height_fact
            * mass_scaler
            / GRAV_ENGLISH_LBM
        )


class BWBFuselageMass(om.ExplicitComponent):
    """
    Computes the mass of the fuselage for BWB aircraft. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')

        add_aviary_output(self, Aircraft.Fuselage.MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(Aircraft.Fuselage.MASS, '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        cabin_area = inputs[Aircraft.Fuselage.CABIN_AREA]
        if gross_weight <= 0.0:
            if verbosity > Verbosity.BRIEF:
                raise om.AnalysisError('Mission.Design.GROSS_MASS must be positive.')

        outputs[Aircraft.Fuselage.MASS] = 1.8 * gross_weight**0.167 * cabin_area**1.06

    def compute_partials(self, inputs, J):
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        cabin_area = inputs[Aircraft.Fuselage.CABIN_AREA]
        J[Aircraft.Fuselage.MASS, Mission.Design.GROSS_MASS] = (
            0.167 * 1.8 * gross_weight**-0.833 * cabin_area**1.06
        )
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.CABIN_AREA] = (
            1.06 * 1.8 * gross_weight**0.167 * cabin_area**0.06
        )


class BWBAftBodyMass(om.ExplicitComponent):
    """Mass of aft body for BWB aircraft"""

    def initialize(self):
        add_aviary_option(self, Settings.VERBOSITY)
        add_aviary_option(self, Aircraft.Engine.NUM_FUSELAGE_ENGINES)

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuselage.PLANFORM_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.CABIN_AREA, units='ft**2')
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='ft')
        add_aviary_input(self, Aircraft.Wing.ROOT_CHORD, units='ft')
        add_aviary_input(self, Aircraft.Wing.COMPOSITE_FRACTION, units='unitless')
        self.add_input(
            'Rear_spar_percent_chord',
            0.7,
            units='unitless',
            desc='RSPSOB: Rear spar percent chord for BWB at side of body',
        )
        self.add_input(
            'Rear_spar_percent_chord_centerline',
            0.7,
            units='unitless',
            desc='RSPCHD: Rear spar percent chord for BWB at fuselage centerline',
        )

        add_aviary_output(self, Aircraft.Fuselage.AFTBODY_MASS, units='lbm')
        add_aviary_output(self, Aircraft.Wing.BWB_AFTBODY_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Fuselage.AFTBODY_MASS,
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.ROOT_CHORD,
                'Rear_spar_percent_chord',
                'Rear_spar_percent_chord_centerline',
            ],
        )
        self.declare_partials(
            Aircraft.Wing.BWB_AFTBODY_MASS,
            [
                Mission.Design.GROSS_MASS,
                Aircraft.Fuselage.PLANFORM_AREA,
                Aircraft.Fuselage.CABIN_AREA,
                Aircraft.Fuselage.LENGTH,
                Aircraft.Wing.ROOT_CHORD,
                Aircraft.Wing.COMPOSITE_FRACTION,
                'Rear_spar_percent_chord',
                'Rear_spar_percent_chord_centerline',
            ],
        )

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        num_fuse_eng = self.options[Aircraft.Engine.NUM_FUSELAGE_ENGINES]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fuse_area = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        cabin_area = inputs[Aircraft.Fuselage.CABIN_AREA]
        length = inputs[Aircraft.Fuselage.LENGTH]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD]
        rear_spar_percent_chord = inputs['Rear_spar_percent_chord']
        rear_spar_percent_chord_centerline = inputs['Rear_spar_percent_chord_centerline']
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]

        if rear_spar_percent_chord <= 0.0 or rear_spar_percent_chord >= 1.0:
            if verbosity > Verbosity.BRIEF:
                raise ValueError('Rear_spar_percent_chord must be within 0 and 1.')
        if rear_spar_percent_chord_centerline <= 0.0 or rear_spar_percent_chord_centerline >= 1.0:
            if verbosity > Verbosity.BRIEF:
                raise ValueError('Rear_spar_percent_chord_centerline must be within 0 and 1.')
        if length <= 0.0:
            if verbosity > Verbosity.BRIEF:
                raise ValueError('Aircraft.Fuselage.LENGTH must be positive.')

        aftbody_area = fuse_area - cabin_area
        aftbody_tr = ((1.0 - rear_spar_percent_chord) * root_chord / rear_spar_percent_chord) / (
            (1.0 - rear_spar_percent_chord_centerline) * length
        )
        aftbody_weight = (
            (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**0.2
            * (0.5 + aftbody_tr)
        )
        aftbody_weight_adjusted = aftbody_weight * (1.0 - 0.17 * comp_frac)
        outputs[Aircraft.Fuselage.AFTBODY_MASS] = aftbody_weight / GRAV_ENGLISH_LBM
        outputs[Aircraft.Wing.BWB_AFTBODY_MASS] = aftbody_weight_adjusted / GRAV_ENGLISH_LBM

    def compute_partials(self, inputs, J):
        num_fuse_eng = self.options[Aircraft.Engine.NUM_FUSELAGE_ENGINES]
        gross_weight = inputs[Mission.Design.GROSS_MASS] * GRAV_ENGLISH_LBM
        fuse_area = inputs[Aircraft.Fuselage.PLANFORM_AREA]
        cabin_area = inputs[Aircraft.Fuselage.CABIN_AREA]
        length = inputs[Aircraft.Fuselage.LENGTH]
        root_chord = inputs[Aircraft.Wing.ROOT_CHORD]
        rear_spar_percent_chord = inputs['Rear_spar_percent_chord']
        rear_spar_percent_chord_centerline = inputs['Rear_spar_percent_chord_centerline']
        comp_frac = inputs[Aircraft.Wing.COMPOSITE_FRACTION]
        fac = 1.0 - 0.17 * comp_frac

        aftbody_area = fuse_area - cabin_area
        aftbody_tr = ((1.0 - rear_spar_percent_chord) * root_chord / rear_spar_percent_chord) / (
            (1.0 - rear_spar_percent_chord_centerline) * length
        )
        aftbody_weight = (
            (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**0.2
            * (0.5 + aftbody_tr)
        )

        J[Aircraft.Fuselage.AFTBODY_MASS, Mission.Design.GROSS_MASS] = (
            0.2
            * (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**-0.8
            * (0.5 + aftbody_tr)
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, Mission.Design.GROSS_MASS] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, Mission.Design.GROSS_MASS] * fac
        )
        J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Fuselage.PLANFORM_AREA] = (
            (1.0 + 0.05 * num_fuse_eng) * 0.53 * gross_weight**0.2 * (0.5 + aftbody_tr)
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, Aircraft.Fuselage.PLANFORM_AREA] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Fuselage.PLANFORM_AREA] * fac
        )
        J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Fuselage.CABIN_AREA] = (
            -(1.0 + 0.05 * num_fuse_eng) * 0.53 * gross_weight**0.2 * (0.5 + aftbody_tr)
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, Aircraft.Fuselage.CABIN_AREA] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Fuselage.CABIN_AREA] * fac
        )
        daftbody_tr_droot_chord = ((1.0 - rear_spar_percent_chord) / rear_spar_percent_chord) / (
            (1.0 - rear_spar_percent_chord_centerline) * length
        )
        J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Wing.ROOT_CHORD] = (
            (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**0.2
            * daftbody_tr_droot_chord
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, Aircraft.Wing.ROOT_CHORD] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Wing.ROOT_CHORD] * fac
        )
        daftbody_tr_dlength = -(
            (1.0 - rear_spar_percent_chord) * root_chord / rear_spar_percent_chord
        ) / ((1.0 - rear_spar_percent_chord_centerline) * length**2)
        J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Fuselage.LENGTH] = (
            (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**0.2
            * daftbody_tr_dlength
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, Aircraft.Fuselage.LENGTH] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, Aircraft.Fuselage.LENGTH] * fac
        )
        daftbody_tr_drspc = (
            -1.0
            / rear_spar_percent_chord**2
            * root_chord
            / ((1.0 - rear_spar_percent_chord_centerline) * length)
        )
        J[Aircraft.Fuselage.AFTBODY_MASS, 'Rear_spar_percent_chord'] = (
            (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**0.2
            * daftbody_tr_drspc
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, 'Rear_spar_percent_chord'] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, 'Rear_spar_percent_chord'] * fac
        )
        daftbody_tr_drspcc = (
            (1.0 - rear_spar_percent_chord) * root_chord / rear_spar_percent_chord
        ) / ((1.0 - rear_spar_percent_chord_centerline) ** 2 * length)
        J[Aircraft.Fuselage.AFTBODY_MASS, 'Rear_spar_percent_chord_centerline'] = (
            (1.0 + 0.05 * num_fuse_eng)
            * 0.53
            * aftbody_area
            * gross_weight**0.2
            * daftbody_tr_drspcc
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, 'Rear_spar_percent_chord_centerline'] = (
            J[Aircraft.Fuselage.AFTBODY_MASS, 'Rear_spar_percent_chord_centerline'] * fac
        )
        J[Aircraft.Wing.BWB_AFTBODY_MASS, Aircraft.Wing.COMPOSITE_FRACTION] = -0.17 * aftbody_weight
