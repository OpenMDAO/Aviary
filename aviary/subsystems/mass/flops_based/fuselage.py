import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.subsystems.mass.flops_based.distributed_prop import \
    distributed_engine_count_factor
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class TransportFuselageMass(om.ExplicitComponent):
    """
    Computes the mass of the fuselage for transports. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, val=0.0)

        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER, val=1.0)

        add_aviary_input(self, Aircraft.Fuselage.AVG_DIAMETER, val=0.0)

        add_aviary_output(self, Aircraft.Fuselage.MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Fuselage.MASS, "*")

    def compute(self, inputs, outputs):
        aviary_options: AviaryValues = self.options['aviary_options']
        length = inputs[Aircraft.Fuselage.LENGTH]
        scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        avg_diameter = inputs[Aircraft.Fuselage.AVG_DIAMETER]

        num_fuse = aviary_options.get_val(Aircraft.Fuselage.NUM_FUSELAGES)
        num_fuse_eng = aviary_options.get_val(
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)
        military_cargo = aviary_options.get_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR)

        mil_factor = 1.38 if military_cargo else 1.0

        outputs[Aircraft.Fuselage.MASS] = (
            scaler * 1.35 * (avg_diameter * length) ** 1.28 * (1.0 + 0.05 * num_fuse_eng_fact) *
            mil_factor * num_fuse / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        aviary_options: AviaryValues = self.options['aviary_options']
        length = inputs[Aircraft.Fuselage.LENGTH]
        scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        avg_diameter = inputs[Aircraft.Fuselage.AVG_DIAMETER]
        num_fuse = aviary_options.get_val(Aircraft.Fuselage.NUM_FUSELAGES)
        num_fuse_eng = aviary_options.get_val(
            Aircraft.Propulsion.TOTAL_NUM_FUSELAGE_ENGINES)
        num_fuse_eng_fact = distributed_engine_count_factor(num_fuse_eng)
        military_cargo = aviary_options.get_val(Aircraft.Fuselage.MILITARY_CARGO_FLOOR)

        # avg_diameter = (max_height + max_width) / 2.
        avg_diameter_exp = avg_diameter ** 1.28
        length_exp = length ** 1.28
        height_width_exp = (avg_diameter)**0.28
        mil_factor = 1.38 if military_cargo else 1.0
        addtl_factor = (1.0 + 0.05 * num_fuse_eng_fact) * mil_factor * num_fuse

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_SCALER] = 1.35 * \
            avg_diameter_exp * length_exp * addtl_factor / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.LENGTH] = scaler * \
            1.728 * avg_diameter_exp * length ** 0.28 * addtl_factor / GRAV_ENGLISH_LBM
        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_DIAMETER] = scaler * \
            1.728 * length_exp * height_width_exp * addtl_factor / GRAV_ENGLISH_LBM


class AltFuselageMass(om.ExplicitComponent):
    """
    Computes the mass of the fuselage using the alternate method. The methodology
    is based on the FLOPS weight equations, modified to output mass instead
    of weight.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Fuselage.MASS_SCALER, 1.0)

        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, 0.0)

        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, 0.0)

        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, 0.0)

        add_aviary_output(self, Aircraft.Fuselage.MASS, 1.0)

    def setup_partials(self):
        self.declare_partials(of='*', wrt='*')

    def compute(self, inputs, outputs):
        mass_scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        fuse_wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]
        fuse_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        fuse_width = inputs[Aircraft.Fuselage.MAX_WIDTH]

        outputs[Aircraft.Fuselage.MASS] = (
            3.939 * fuse_wetted_area /
            (fuse_height / fuse_width)**0.221 * mass_scaler / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        mass_scaler = inputs[Aircraft.Fuselage.MASS_SCALER]
        fuse_wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]
        fuse_height = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        fuse_width = inputs[Aircraft.Fuselage.MAX_WIDTH]
        fuse_height_fact = fuse_height ** 0.221
        fuse_width_fact = fuse_width ** 0.221
        total_fact = fuse_height_fact / fuse_width_fact

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MASS_SCALER] = (
            3.939 * fuse_wetted_area / total_fact / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = (
            3.939 / total_fact * mass_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MAX_HEIGHT] = (
            -0.870519 * fuse_wetted_area * fuse_width_fact
            * fuse_height**-1.221 * mass_scaler / GRAV_ENGLISH_LBM
        )

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.MAX_WIDTH] = (
            0.870519 * fuse_wetted_area * fuse_width**-0.779 /
            fuse_height_fact * mass_scaler / GRAV_ENGLISH_LBM
        )
