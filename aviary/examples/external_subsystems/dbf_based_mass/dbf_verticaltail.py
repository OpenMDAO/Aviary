import numpy as np
import os

import openmdao.api as om
from openmdao.utils.cs_safe import abs as cs_abs

from aviary.examples.external_subsystems.dbf_based_mass.materials_database import materials
from aviary.utils.utils import wrapped_convert_units
from aviary.variable_info.functions import add_aviary_input, add_aviary_output, add_aviary_option
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

    # Capture units locally so it's always available
    def set_func(meta_unused, val):
        # val might be tuple or raw value
        if isinstance(val, tuple):
            return wrapped_convert_units(val, units)
        else:
            return wrapped_convert_units((val, units), units)

    return {
        'name': var_key,
        'default': (default_val, units),
        'set_function': set_func,
        'desc': desc,
    }

class DBFVerticalTailMass(om.ExplicitComponent):
    def initialize(self):
        self.options.declare(Aircraft.VerticalTail.Dbf.AIRFOIL_PATH, types=str, allow_none=False)
        self.options.declare(Aircraft.VerticalTail.Dbf.RIB_MATERIALS, types=(list,))

        # Declare options using Aircraft.VerticalTail.Dbf metadata keys
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.NUM_SPARS, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SPAR_OUTER_DIAMETER, 'm'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SPAR_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SPAR_WALL_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.RIB_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.RIB_LIGHTENING_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SKIN_DENSITY, 'kg/m**2'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.GLUE_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.STRINGER_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.STRINGER_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SHEETING_THICKNESS, 'm'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SHEETING_DENSITY, 'kg/m**3'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SHEETING_COVERAGE, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.SHEETING_LIGHTENING_FACTOR, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.NUM_STRINGERS, 'unitless'))
        self.options.declare(**make_units_option(Aircraft.VerticalTail.Dbf.MISC_MASS, 'kg'))

    def setup(self):
        # Still user inputs:
        add_aviary_input(self, Aircraft.VerticalTail.SPAN, units='m')
        add_aviary_input(self, Aircraft.VerticalTail.ROOT_CHORD, units='m')
        add_aviary_input(self, Aircraft.VerticalTail.WETTED_AREA, units='m**2')

        add_aviary_output(self, Aircraft.VerticalTail.MASS, units='kg')

    def setup_partials(self):
        self.declare_partials(
            of=Aircraft.VerticalTail.MASS,
            wrt=[
                Aircraft.VerticalTail.SPAN,
                Aircraft.VerticalTail.ROOT_CHORD,
                Aircraft.VerticalTail.WETTED_AREA,
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
            The option key (usually Aircraft.VerticalTail.Dbf.*).
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


    def load_airfoil_csv(self, file_path, delimiter=',', header=False):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Airfoil CSV file '{file_path}' not found.")

        skip = 1 if header else 0
        data = np.loadtxt(file_path, delimiter=delimiter, skiprows=skip)

        if data.shape[1] < 2:
            raise ValueError('CSV must contain at least two columns for x and y coordinates.')

        x = data[:, 0]
        y = data[:, 1]

        x_min = np.min(x)
        x_max = np.max(x)
        chord_length = x_max - x_min

        if chord_length <= 0:
            raise ValueError('Invalid airfoil: chord length must be > 0.')

        x_normalized = (x - x_min) / chord_length
        y_normalized = y / chord_length

        return x_normalized, y_normalized

    def shoelace_area(self, x, y):
        return 0.5 * cs_abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    def compute(self, inputs, outputs):
        # From inputs
        span = inputs[Aircraft.VerticalTail.SPAN]
        chord = inputs[Aircraft.VerticalTail.ROOT_CHORD]
        wetted_area = inputs[Aircraft.VerticalTail.WETTED_AREA]

        if span <= 0:
            raise ValueError(f'VerticalTail span must be > 0, got {span}')
        if chord <= 0:
            raise ValueError(f'Root chord must be > 0, got {chord}')
        if wetted_area <= 0:
            raise ValueError(f'Wetted area must be > 0, got {wetted_area}')

        # From options
        num_spars = self.options[Aircraft.VerticalTail.Dbf.NUM_SPARS]
        rib_lightening_factor = self.options[Aircraft.VerticalTail.Dbf.RIB_LIGHTENING_FACTOR]
        rib_thickness = self.options[Aircraft.VerticalTail.Dbf.RIB_THICKNESS]
        rho_skin = self.options[Aircraft.VerticalTail.Dbf.SKIN_DENSITY]
        spar_outer_diameter = self.options[Aircraft.VerticalTail.Dbf.SPAR_OUTER_DIAMETER]
        rho_spar = self.options[Aircraft.VerticalTail.Dbf.SPAR_DENSITY]
        spar_wall_thickness = self.options[Aircraft.VerticalTail.Dbf.SPAR_WALL_THICKNESS]
        glue_factor = self.options[Aircraft.VerticalTail.Dbf.GLUE_FACTOR]
        stringer_thickness = self.options[Aircraft.VerticalTail.Dbf.STRINGER_THICKNESS]
        rho_stringer = self.options[Aircraft.VerticalTail.Dbf.STRINGER_DENSITY]
        sheeting_thickness = self.options[Aircraft.VerticalTail.Dbf.SHEETING_THICKNESS]
        sheeting_coverage = self.options[Aircraft.VerticalTail.Dbf.SHEETING_COVERAGE]
        rho_sheeting = self.options[Aircraft.VerticalTail.Dbf.SHEETING_DENSITY]
        sheeting_lightening_factor = self.options[Aircraft.VerticalTail.Dbf.SHEETING_LIGHTENING_FACTOR]
        num_stringer = self.options[Aircraft.VerticalTail.Dbf.NUM_STRINGERS]
        rib_materials = self.options[Aircraft.VerticalTail.Dbf.RIB_MATERIALS]  # stays string key
        airfoil_data_file = self.options[Aircraft.VerticalTail.Dbf.AIRFOIL_PATH]
        misc_mass = self.options[Aircraft.VerticalTail.Dbf.MISC_MASS]

        if len(rib_materials) != len(rib_thickness):
            raise ValueError(
                f'Length mismatch: {len(rib_materials)} rib_materials vs '
                f'{len(rib_thickness)} rib_thicknesses. These must match.'
            )

        rho_rib = np.array([(materials.get_item(m)[0]) for m in rib_materials])

        x_coords, y_coords = self.load_airfoil_csv(airfoil_data_file, header=True)
        n_area = self.shoelace_area(x_coords, y_coords)

        if n_area < 0.01:
            raise ValueError(
                f'Computed normalized airfoil area is suspiciously small: {n_area:.5f}'
            )

        cs_area = n_area * (chord**2) * rib_lightening_factor

        rib_volumes = cs_area * rib_thickness
        spar_volume = (
            num_spars
            * span
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        )
        sheeting_volume = (
            wetted_area * sheeting_coverage * sheeting_lightening_factor * sheeting_thickness
        )
        stringer_volume = stringer_thickness**2 * num_stringer * span

        rib_mass = np.sum(rib_volumes * rho_rib)
        sheeting_mass = sheeting_volume * rho_sheeting
        stringer_mass = stringer_volume * rho_stringer
        spar_mass = spar_volume * rho_spar
        skin_mass = rho_skin * wetted_area

        structural_mass = stringer_mass + sheeting_mass + rib_mass + spar_mass + skin_mass
        total_mass = (1 + glue_factor) * structural_mass + misc_mass

        outputs[Aircraft.VerticalTail.MASS] = total_mass

    def compute_partials(self, inputs, J):
        # From inputs
        chord = inputs[Aircraft.VerticalTail.ROOT_CHORD]

        # From options
        num_spars = self.options[Aircraft.VerticalTail.Dbf.NUM_SPARS]
        rib_lightening_factor = self.options[Aircraft.VerticalTail.Dbf.RIB_LIGHTENING_FACTOR]
        rib_thickness = self.options[Aircraft.VerticalTail.Dbf.RIB_THICKNESS]
        rho_skin = self.options[Aircraft.VerticalTail.Dbf.SKIN_DENSITY]
        spar_outer_diameter = self.options[Aircraft.VerticalTail.Dbf.SPAR_OUTER_DIAMETER]
        rho_spar = self.options[Aircraft.VerticalTail.Dbf.SPAR_DENSITY]
        spar_wall_thickness = self.options[Aircraft.VerticalTail.Dbf.SPAR_WALL_THICKNESS]
        glue_factor = self.options[Aircraft.VerticalTail.Dbf.GLUE_FACTOR]
        stringer_thickness = self.options[Aircraft.VerticalTail.Dbf.STRINGER_THICKNESS]
        rho_stringer = self.options[Aircraft.VerticalTail.Dbf.STRINGER_DENSITY]
        sheeting_thickness = self.options[Aircraft.VerticalTail.Dbf.SHEETING_THICKNESS]
        sheeting_coverage = self.options[Aircraft.VerticalTail.Dbf.SHEETING_COVERAGE]
        rho_sheeting = self.options[Aircraft.VerticalTail.Dbf.SHEETING_DENSITY]
        sheeting_lightening_factor = self.options[Aircraft.VerticalTail.Dbf.SHEETING_LIGHTENING_FACTOR]
        num_stringer = self.options[Aircraft.VerticalTail.Dbf.NUM_STRINGERS]
        rib_materials = self.options[Aircraft.VerticalTail.Dbf.RIB_MATERIALS]  # stays string key
        airfoil_data_file = self.options[Aircraft.VerticalTail.Dbf.AIRFOIL_PATH]

        rho_rib = np.array([(materials.get_item(m)[0]) for m in rib_materials])

        x_coords, y_coords = self.load_airfoil_csv(airfoil_data_file, header=True)
        n_area = self.shoelace_area(x_coords, y_coords)

        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.SPAN] = (
            num_stringer * rho_stringer * stringer_thickness**2
            + num_spars
            * rho_spar
            * np.pi
            * (spar_outer_diameter * spar_wall_thickness - spar_wall_thickness**2)
        ) * (1 + glue_factor)

        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.WETTED_AREA] = (
            rho_skin
            + (sheeting_coverage * sheeting_lightening_factor * sheeting_thickness * rho_sheeting)
        ) * (1 + glue_factor)

        J[Aircraft.VerticalTail.MASS, Aircraft.VerticalTail.ROOT_CHORD] = (
            2 * chord * rib_lightening_factor * n_area * np.sum(rib_thickness * rho_rib)
        ) * (1 + glue_factor)


if __name__ == '__main__':
    prob = om.Problem()

    prob.model.add_subsystem(
        'dbf_vtail', DBFVerticalTailMass(), promotes_inputs=['*'], promotes_outputs=['*']
    )

    ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
    rib_thicks = np.where(ribs != 0, 0.125, 0.125)

    # Set required options
    vtail = prob.model.dbf_vtail
    vtail.options[Aircraft.VerticalTail.Dbf.RIB_MATERIALS] = rib_materials
    vtail.options[Aircraft.VerticalTail.Dbf.AIRFOIL_PATH] = (
        r'aviary\examples\external_subsystems\dbf_based_mass\mh84-il.csv'
    )
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_COVERAGE, val=0.4, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_LIGHTENING_FACTOR, val=1, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    vtail.set_option(Aircraft.VerticalTail.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.NUM_STRINGERS, val=2.5, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.GLUE_FACTOR, val=0.15, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.NUM_SPARS, val=1.1, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.RIB_LIGHTENING_FACTOR, val=2/3, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.RIB_THICKNESS, val=rib_thicks, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SPAR_DENSITY, val=2, units='g/cm**3')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SPAR_OUTER_DIAMETER, val=1, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SPAR_WALL_THICKNESS, val=0.0625, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.MISC_MASS, val=0.0, units='kg')

    # Setup problem with constant above options
    prob.setup()

    # Set values for aero-driving variables
    prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, val=20, units='inch')
    prob.set_val(Aircraft.VerticalTail.SPAN, val=4.667, units='ft')
    prob.set_val(Aircraft.VerticalTail.WETTED_AREA, val=0.85, units='m**2')

    prob.run_model()

    total_mass = prob.get_val(Aircraft.VerticalTail.MASS)
    print(f'Total mass of the dbf vtail: {float(total_mass):.3f} kg')
