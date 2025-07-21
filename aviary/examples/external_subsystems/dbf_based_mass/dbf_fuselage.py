import numpy as np

import openmdao.api as om

from aviary.examples.external_subsystems.dbf_based_mass.materials_database import materials
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.variable_meta_data import _MetaData


def make_units_option(var_key, units=None, default_val=None, desc=None, meta_data=_MetaData):
    meta = meta_data[var_key]

    default_units = meta['units']

    if units is None:
        units = default_units
    if desc is None:
        desc = meta['desc']
    if default_val is None:
        default_val = meta['default_value']

    return {
        'name': var_key,
        'default': (default_val, units),
        'set_function': lambda meta, val: (wrapped_convert_units(val, units), units),
        'desc': desc,
    }


class DBFFuselageMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('bulkhead_materials', types=(list,))

        # Declare options using Aircraft.Fuselage.Dbf metadata keys
        # Note: These may need to be adjusted based on actual metadata keys available
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.NUM_SPARS, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SPAR_OUTER_DIAMETER, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SPAR_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SPAR_WALL_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.BULKHEAD_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.BULKHEAD_LIGHTENING_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SKIN_DENSITY, 'kg/m**2'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.FLOOR_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.FLOOR_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.FLOOR_LENGTH, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.GLUE_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.STRINGER_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.STRINGER_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SHEETING_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SHEETING_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SHEETING_COVERAGE, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.SHEETING_LIGHTENING_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.Fuselage.Dbf.MISC_MASS, 'kg'))

    def setup(self):
        # Required geometry inputs
        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='m')
        add_aviary_input(self, Aircraft.Fuselage.AVG_HEIGHT, units='m')
        add_aviary_input(self, Aircraft.Fuselage.WETTED_AREA, units='m**2')
        add_aviary_input(self, Aircraft.Fuselage.AVG_WIDTH, units='m')

        add_aviary_output(
            self,
            Aircraft.Fuselage.MASS,
            units='kg',
        )

    def setup_partials(self):
        self.declare_partials(
            of=Aircraft.Fuselage.MASS,
            wrt=[
                Aircraft.Fuselage.LENGTH,
                Aircraft.Fuselage.AVG_HEIGHT,
                Aircraft.Fuselage.WETTED_AREA,
                Aircraft.Fuselage.AVG_WIDTH,
            ],
        )

    def set_option(self, option_key, val=None, units=None):
        """
        Helper to set an OpenMDAO component option with units.

        Parameters
        ----------
        comp : om.Component
            The component instance owning the option.
        option_key : str or key
            The option key (usually Aircraft.Fuselage.Dbf.*).
        val : any
            The value to set.
        units : str or None
            Optional units string. If None, attempts to get default units from metadata.
        """
        from aviary.variable_info.variable_meta_data import _MetaData

        # Get default units from metadata if none given
        if units is None and option_key in _MetaData:
            units = _MetaData[option_key]['units']
        if val is None and option_key in _MetaData:
            val = _MetaData[option_key]['default_value']

        self.options[option_key] = (val, units)

    def compute(self, inputs, outputs):
        # Inputs
        length = inputs[Aircraft.Fuselage.LENGTH]
        height = inputs[Aircraft.Fuselage.AVG_HEIGHT]
        width = inputs[Aircraft.Fuselage.AVG_WIDTH]
        wetted_area = inputs[Aircraft.Fuselage.WETTED_AREA]

        # From options
        num_spars = self.options[Aircraft.Fuselage.Dbf.NUM_SPARS][0]
        spar_outer_diameter = self.options[Aircraft.Fuselage.Dbf.SPAR_OUTER_DIAMETER][0]
        rho_spar = self.options[Aircraft.Fuselage.Dbf.SPAR_DENSITY][0]
        spar_wall_thickness = self.options[Aircraft.Fuselage.Dbf.SPAR_WALL_THICKNESS][0]
        bulkhead_thickness = self.options[Aircraft.Fuselage.Dbf.BULKHEAD_THICKNESS][0]
        bulkhead_lightening_factor = self.options[Aircraft.Fuselage.Dbf.BULKHEAD_LIGHTENING_FACTOR][0]
        rho_skin = self.options[Aircraft.Fuselage.Dbf.SKIN_DENSITY][0]
        rho_floor = self.options[Aircraft.Fuselage.Dbf.FLOOR_DENSITY][0]
        floor_thickness = self.options[Aircraft.Fuselage.Dbf.FLOOR_THICKNESS][0]
        floor_length = self.options[Aircraft.Fuselage.Dbf.FLOOR_LENGTH][0]
        glue_factor = self.options[Aircraft.Fuselage.Dbf.GLUE_FACTOR][0]
        stringer_thickness = self.options[Aircraft.Fuselage.Dbf.STRINGER_THICKNESS][0]
        rho_stringer = self.options[Aircraft.Fuselage.Dbf.STRINGER_DENSITY][0]
        sheeting_thick = self.options[Aircraft.Fuselage.Dbf.SHEETING_THICKNESS][0]
        sheeting_coverage = self.options[Aircraft.Fuselage.Dbf.SHEETING_COVERAGE][0]
        rho_sheeting = self.options[Aircraft.Fuselage.Dbf.SHEETING_DENSITY][0]
        sheeting_lightening_factor = self.options[Aircraft.Fuselage.Dbf.SHEETING_LIGHTENING_FACTOR][0]
        bulkhead_materials = self.options['bulkhead_materials']
        misc_mass = self.options[Aircraft.Fuselage.Dbf.MISC_MASS][0]

        rho_rib = np.array([(materials.get_item(m)[0]) for m in bulkhead_materials])
        cs_area = width * height * bulkhead_lightening_factor
        rib_volumes = cs_area * bulkhead_thickness
        rib_mass = np.sum(rib_volumes * rho_rib)

        # Spar volume
        spar_volume = (
            num_spars
            * length
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        )

        # Other volumes
        sheeting_volume = (
            wetted_area * sheeting_coverage * sheeting_lightening_factor * sheeting_thick
        )
        stringer_volume = 4 * stringer_thickness**2 * (length + width + height)

        # Mass calculations
        sheeting_mass = sheeting_volume * rho_sheeting
        stringer_mass = stringer_volume * rho_stringer
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area
        floor_mass = rho_floor * floor_length * width * floor_thickness

        # Total structural mass
        structural_mass = (
            rib_mass + spar_mass + floor_mass + stringer_mass + sheeting_mass + skin_mass
        )

        total_mass = structural_mass * (1 + glue_factor) + misc_mass

        outputs[Aircraft.Fuselage.MASS] = total_mass

    def compute_partials(self, inputs, J):
        # Inputs
        height = inputs[Aircraft.Fuselage.AVG_HEIGHT]
        width = inputs[Aircraft.Fuselage.AVG_WIDTH]

        # From options
        num_spars = self.options[Aircraft.Fuselage.Dbf.NUM_SPARS][0]
        spar_outer_diameter = self.options[Aircraft.Fuselage.Dbf.SPAR_OUTER_DIAMETER][0]
        rho_spar = self.options[Aircraft.Fuselage.Dbf.SPAR_DENSITY][0]
        spar_wall_thickness = self.options[Aircraft.Fuselage.Dbf.SPAR_WALL_THICKNESS][0]
        bulkhead_thickness = self.options[Aircraft.Fuselage.Dbf.BULKHEAD_THICKNESS][0]
        bulkhead_lightening_factor = self.options[Aircraft.Fuselage.Dbf.BULKHEAD_LIGHTENING_FACTOR][0]
        rho_skin = self.options[Aircraft.Fuselage.Dbf.SKIN_DENSITY][0]
        rho_floor = self.options[Aircraft.Fuselage.Dbf.FLOOR_DENSITY][0]
        floor_thickness = self.options[Aircraft.Fuselage.Dbf.FLOOR_THICKNESS][0]
        floor_length = self.options[Aircraft.Fuselage.Dbf.FLOOR_LENGTH][0]
        glue_factor = self.options[Aircraft.Fuselage.Dbf.GLUE_FACTOR][0]
        stringer_thickness = self.options[Aircraft.Fuselage.Dbf.STRINGER_THICKNESS][0]
        rho_stringer = self.options[Aircraft.Fuselage.Dbf.STRINGER_DENSITY][0]
        sheeting_thick = self.options[Aircraft.Fuselage.Dbf.SHEETING_THICKNESS][0]
        sheeting_coverage = self.options[Aircraft.Fuselage.Dbf.SHEETING_COVERAGE][0]
        rho_sheeting = self.options[Aircraft.Fuselage.Dbf.SHEETING_DENSITY][0]
        sheeting_lightening_factor = self.options[Aircraft.Fuselage.Dbf.SHEETING_LIGHTENING_FACTOR][0]
        bulkhead_materials = self.options['bulkhead_materials']

        rho_rib = np.array([(materials.get_item(m)[0]) for m in bulkhead_materials])

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_HEIGHT] = (
            4 * stringer_thickness**2 * rho_stringer
            + width * bulkhead_lightening_factor * np.sum(rho_rib * bulkhead_thickness)
        ) * (1 + glue_factor)

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.AVG_WIDTH] = (
            4 * stringer_thickness**2 * rho_stringer
            + floor_length * floor_thickness * rho_floor
            + height * bulkhead_lightening_factor * np.sum(rho_rib * bulkhead_thickness)
        ) * (1 + glue_factor)

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.LENGTH] = (
            num_spars
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
            * rho_spar
            + 4 * stringer_thickness**2 * rho_stringer
        ) * (1 + glue_factor)

        J[Aircraft.Fuselage.MASS, Aircraft.Fuselage.WETTED_AREA] = (
            rho_skin
            + sheeting_coverage * sheeting_lightening_factor * sheeting_thick * rho_sheeting
        ) * (1 + glue_factor)


if __name__ == '__main__':
    prob = om.Problem()

    # === Prepare values before setup ===
    ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2])
    bulkhead_materials = np.where(ribs != 0, 'Ply', 'Balsa').tolist()
    rib_thicks = np.where(ribs == 2, 0.25, 0.125)

    prob.model.add_subsystem(
        'dbf_fuselage', DBFFuselageMass(), promotes_inputs=['*'], promotes_outputs=['*']
    )

    # Set required options
    fuselage = prob.model.dbf_fuselage
    fuselage.options['bulkhead_materials'] = bulkhead_materials
    fuselage.set_option(Aircraft.Fuselage.Dbf.BULKHEAD_THICKNESS, val=rib_thicks, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.NUM_SPARS, val=0.5, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.BULKHEAD_LIGHTENING_FACTOR, val=0.18, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_COVERAGE, val=1, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_LIGHTENING_FACTOR, val=0.3, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.GLUE_FACTOR, val=0.08, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.FLOOR_LENGTH, val=2, units='ft')
    fuselage.set_option(Aircraft.Fuselage.Dbf.FLOOR_DENSITY, val=340, units='kg/m**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.FLOOR_THICKNESS, val=0.125, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SPAR_DENSITY, val=2, units='g/cm**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SPAR_OUTER_DIAMETER, val=1, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SPAR_WALL_THICKNESS, val=0.0625, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.MISC_MASS, val=0.0, units='kg')

    # Setup problem with constant above options
    prob.setup()

    prob.set_val(Aircraft.Fuselage.LENGTH, val=4, units='ft')
    prob.set_val(Aircraft.Fuselage.AVG_HEIGHT, val=5, units='inch')
    prob.set_val(Aircraft.Fuselage.AVG_WIDTH, val=4, units='inch')
    prob.set_val(Aircraft.Fuselage.WETTED_AREA, val=904, units='inch**2')

    prob.run_model()

    total_mass = prob.get_val(Aircraft.Fuselage.MASS)
    print(f'Total mass of the dbf fuselage: {float(total_mass):.3f} kg')