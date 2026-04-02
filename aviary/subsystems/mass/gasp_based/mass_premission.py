import openmdao.api as om

from aviary.subsystems.mass.gasp_based.design_load import BWBDesignLoadGroup, DesignLoadGroup
from aviary.subsystems.mass.gasp_based.equipment_and_useful_load import EquipAndUsefulLoadMassGroup
from aviary.subsystems.mass.gasp_based.fixed import FixedMassGroup
from aviary.subsystems.mass.gasp_based.fuel import (
    BodyTankCalculations,
    BWBFuselageMass,
    FuelComponents,
    FuelMass,
    FuelSysAndFullFuselageMass,
    FuselageMass,
)
from aviary.subsystems.mass.gasp_based.mass_summation import MassSummation
from aviary.subsystems.mass.gasp_based.wing import BWBWingMassGroup, WingMassGroup
from aviary.variable_info.enums import AircraftTypes
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft


class MassPremission(om.Group):
    """Pre-mission mass group for GASP-based mass."""

    def initialize(self):
        add_aviary_option(self, Aircraft.Design.TYPE)

    def setup(self):
        design_type = self.options[Aircraft.Design.TYPE]

        # # output values from design_load that are connected to fixed_mass via promotion
        # fixed_mass_design_load_values = ['max_mach', 'min_dive_vel']

        # # output values from fixed_mass that are connected to wing_mass via promotion
        # wing_mass_fixed_mass_values = [
        #     'c_strut_braced',
        #     'c_gear_loc',
        #     'half_sweep',
        # ]

        # # output values from design_load that are connected to fuel_mass via promotion
        # fuel_mass_design_load_values = ['min_dive_vel']

        # # output values from fixed_mass that are connected to fuel_mass via promotion
        # fuel_mass_fixed_mass_values = [
        #     'payload_mass_des',
        #     'payload_mass_max',
        #     'wing_mounted_mass',
        #     'eng_comb_mass',
        # ]

        # # combine all necessary inputs and outputs for each group

        # fixed_mass_inputs = fixed_mass_design_load_values + ['density']
        # fixed_mass_outputs = wing_mass_fixed_mass_values + fuel_mass_fixed_mass_values

        # wing_mass_inputs = wing_mass_fixed_mass_values

        # if design_type is AircraftTypes.BLENDED_WING_BODY:
        #     fuel_mass_inputs = fuel_mass_fixed_mass_values
        # elif design_type is AircraftTypes.TRANSPORT:
        #     fuel_mass_inputs = fuel_mass_design_load_values + fuel_mass_fixed_mass_values

        # create the instances of the groups
        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'design_load',
                BWBDesignLoadGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'design_load',
                DesignLoadGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'fixed_mass',
            FixedMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'equip_and_useful_mass',
            EquipAndUsefulLoadMassGroup(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'wing_mass',
                BWBWingMassGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'wing_mass',
                WingMassGroup(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'sys_and_full_fus',
            FuelSysAndFullFuselageMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        if design_type is AircraftTypes.BLENDED_WING_BODY:
            self.add_subsystem(
                'fuselage',
                BWBFuselageMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )
        elif design_type is AircraftTypes.TRANSPORT:
            self.add_subsystem(
                'fuselage',
                FuselageMass(),
                promotes_inputs=['*'],
                promotes_outputs=['*'],
            )

        self.add_subsystem(
            'fuel',
            FuelMass(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'fuel_components',
            FuelComponents(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'body_tank',
            BodyTankCalculations(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'mass_summation',
            MassSummation(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-9
        newton.options['rtol'] = 1e-9
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 10
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 10
        newton.options['err_on_non_converge'] = True
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        newton.options['err_on_non_converge'] = False

        self.linear_solver = om.DirectSolver(assemble_jac=True)
