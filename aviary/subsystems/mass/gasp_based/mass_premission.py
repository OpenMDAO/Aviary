import openmdao.api as om

from aviary.subsystems.mass.gasp_based.design_load import DesignLoadGroup
from aviary.subsystems.mass.gasp_based.equipment_and_useful_load import EquipAndUsefulLoadMass
from aviary.subsystems.mass.gasp_based.fixed import FixedMassGroup
from aviary.subsystems.mass.gasp_based.fuel import FuelMassGroup
from aviary.subsystems.mass.gasp_based.wing import WingMassGroup


class MassPremission(om.Group):
    """Pre-mission mass group for GASP-based mass."""

    def setup(self):
        # output values from design_load that are connected to fixed_mass via promotion
        fixed_mass_design_load_values = ['max_mach', 'min_dive_vel']

        # output values from fixed_mass that are connected to wing_mass via promotion
        wing_mass_fixed_mass_values = [
            'c_strut_braced',
            'c_gear_loc',
            'half_sweep',
        ]

        # output values from design_load that are connected to fuel_mass via promotion
        fuel_mass_design_load_values = ['min_dive_vel']

        # output values from fixed_mass that are connected to fuel_mass via promotion
        fuel_mass_fixed_mass_values = [
            'payload_mass_des',
            'payload_mass_max',
            'wing_mounted_mass',
            'eng_comb_mass',
        ]

        # combine all necessary inputs and outputs for each group

        fixed_mass_inputs = fixed_mass_design_load_values + ['density']
        fixed_mass_outputs = wing_mass_fixed_mass_values + fuel_mass_fixed_mass_values

        wing_mass_inputs = wing_mass_fixed_mass_values

        fuel_mass_inputs = fuel_mass_design_load_values + fuel_mass_fixed_mass_values

        # create the instances of the groups

        self.add_subsystem(
            'design_load',
            DesignLoadGroup(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'fixed_mass',
            FixedMassGroup(),
            promotes_inputs=fixed_mass_inputs + ['aircraft:*', 'mission:*'],
            promotes_outputs=fixed_mass_outputs + ['aircraft:*'],
        )

        self.add_subsystem(
            'equip_and_useful_mass',
            EquipAndUsefulLoadMass(),
            promotes_inputs=['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'wing_mass',
            WingMassGroup(),
            promotes_inputs=wing_mass_inputs + ['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'fuel_mass',
            FuelMassGroup(),
            promotes_inputs=fuel_mass_inputs + ['aircraft:*', 'mission:*'],
            promotes_outputs=['aircraft:*', 'mission:*'],
        )
