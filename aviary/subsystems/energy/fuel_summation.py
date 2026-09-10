import warnings

import openmdao.api as om

from aviary.mission.utils import separate_reserve_phases
from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Aircraft, Mission, Settings


class FuelSummationGroup(om.Group):
    """
    Adds components needed to track aircraft consumption in main and reserve missions, as well
    as add excess fuel capacity constraint if requested.
    """

    def initialize(self):
        # TODO should reserve margin & additional remain options? Or make them full inputs?
        add_aviary_option(self, Mission.RESERVE_FUEL_MARGIN)
        add_aviary_option(self, Mission.RESERVE_FUEL_MASS_ADDITIONAL, units='lbm')
        add_aviary_option(self, Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT)

        add_aviary_option(self, Settings.VERBOSITY)

        self.options.declare(
            'mission_info', types=dict, desc='Mission info dictionary', default=None
        )

    def setup(self):
        # meta_data = self.options['meta_data']
        mission_info = self.options['mission_info']
        verbosity = self.options[Settings.VERBOSITY]

        main_phases, reserve_phases = separate_reserve_phases(mission_info)

        # Check if main_phases[] is accessible
        try:
            main_phases[0]
        except BaseException:
            raise ValueError(
                'main_phases[] dictionary is not accessible. For ENERGY_STATE and '
                'SOLVED_2DOF missions, check_and_preprocess_inputs() must be called '
                'before add_post_mission_systems().'
            )

        # Fuel burn in taxi + takeoff + regular phases
        self.add_subsystem(
            'fuel_burned',
            om.ExecComp(
                'fuel_burned = mass_initial - mass_final',
                mass_initial={'units': 'lbm'},
                mass_final={
                    'units': 'lbm'
                },  # this final mass already includes fuel burned in taxi and takeoff
                fuel_burned={'units': 'lbm'},
            ),
            promotes_inputs=[('mass_initial', Mission.GROSS_MASS)],
            promotes_outputs=[('fuel_burned', Mission.FUEL_MASS)],
        )

        # Fuel burn in reserve phases
        if reserve_phases:
            ecomp = om.ExecComp(
                'reserve_fuel_burned = mass_initial - mass_final',
                mass_initial={'units': 'lbm'},
                mass_final={'units': 'lbm'},
                reserve_fuel_burned={'units': 'lbm'},
            )

            self.add_subsystem(
                'reserve_fuel_burned',
                ecomp,
                promotes=[('reserve_fuel_burned', Mission.RESERVE_FUEL_MASS)],
            )

        reserve_fuel_margin = self.options[Mission.RESERVE_FUEL_MARGIN]
        if reserve_fuel_margin != 0:
            # Originally tried to reference Mission.FUEL_MASS for fuel burn but in some tests this led to errors
            reserve_fuel_frac = om.ExecComp(
                'reserve_fuel_margin_mass = reserve_fuel_margin / 100 * (mass_initial - final_mass)',
                reserve_fuel_margin_mass={'units': 'lbm'},
                reserve_fuel_margin={
                    'units': 'unitless',
                    'val': reserve_fuel_margin,
                },
                mass_initial={'units': 'lbm'},
                final_mass={'units': 'lbm'},
            )

            self.add_subsystem(
                'reserve_fuel_frac',
                reserve_fuel_frac,
                promotes_inputs=[
                    ('mass_initial', Mission.GROSS_MASS),
                    ('reserve_fuel_margin', Mission.RESERVE_FUEL_MARGIN),
                ],
                promotes_outputs=['reserve_fuel_margin_mass'],
            )

        reserve_fuel_mass_additional, units = self.options[Mission.RESERVE_FUEL_MASS_ADDITIONAL]
        reserve_fuel_mass = om.ExecComp(
            'reserve_fuel_mass = reserve_fuel_margin_mass + reserve_fuel_mass_additional + reserve_fuel_burned',
            reserve_fuel_mass={'units': 'lbm', 'shape': 1},
            reserve_fuel_margin_mass={'units': 'lbm', 'val': 0},
            reserve_fuel_mass_additional={'units': 'lbm', 'val': reserve_fuel_mass_additional},
            reserve_fuel_burned={'units': 'lbm', 'val': 0},
        )

        self.add_subsystem(
            'reserve_fuel_mass',
            reserve_fuel_mass,
            promotes_inputs=[
                'reserve_fuel_margin_mass',
                ('reserve_fuel_mass_additional', Mission.RESERVE_FUEL_MASS_ADDITIONAL),
                ('reserve_fuel_burned', Mission.RESERVE_FUEL_MASS),
            ],
            promotes_outputs=[('reserve_fuel_mass', Mission.TOTAL_RESERVE_FUEL_MASS)],
        )

        # Ensure that the usable fuel loaded onto the aircraft is greater or equal to the mission fuel + reserve fuel
        # The aircraft will naturally try to minimize 'total_fuel_mass_constraint' so it's not carrying extra unnecessary fuel
        self.add_subsystem(
            'total_fuel_mass_con',
            om.ExecComp(
                'total_fuel_mass_constraint = total_fuel_mass - mission_fuel_burned - reserve_fuel_mass',
                total_fuel_mass_constraint={'units': 'lbm'},
                total_fuel_mass={'units': 'lbm'},
                mission_fuel_burned={'units': 'lbm'},
                reserve_fuel_mass={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('total_fuel_mass', Mission.TOTAL_FUEL_MASS),
                ('mission_fuel_burned', Mission.FUEL_MASS),
                ('reserve_fuel_mass', Mission.TOTAL_RESERVE_FUEL_MASS),
            ],
            promotes_outputs=[('total_fuel_mass_constraint', Mission.Constraints.MASS_RESIDUAL)],
        )

        ecomp = om.ExecComp(
            'excess_fuel_capacity = total_fuel_capacity - unusable_fuel - overall_fuel',
            total_fuel_capacity={'units': 'lbm'},
            unusable_fuel={'units': 'lbm'},
            overall_fuel={'units': 'lbm'},
            excess_fuel_capacity={'units': 'lbm'},
        )

        self.add_subsystem(
            'excess_fuel_constraint',
            ecomp,
            promotes_inputs=[
                ('total_fuel_capacity', Aircraft.Fuel.MAX_CAPACITY_MASS),
                ('unusable_fuel', Aircraft.Fuel.UNUSABLE_FUEL_MASS),
                ('overall_fuel', Mission.TOTAL_FUEL_MASS),
            ],
            promotes_outputs=[
                ('excess_fuel_capacity', Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY)
            ],
        )

        # determine if the user wants the excess_fuel_capacity constraint active and if so add it to the problem
        if not self.options[Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT]:
            self.add_constraint(
                Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY, lower=0, ref=1.0e5, units='lbm'
            )
        else:
            if verbosity >= Verbosity.BRIEF:
                warnings.warn(
                    'Aircraft.Fuel.IGNORE_FUEL_CAPACITY_CONSTRAINT = True, therefore '
                    'EXCESS_FUEL_MASS_CAPACITY constraint was not added to the Aviary problem. The '
                    'aircraft may not have enough space for fuel, so check the value of '
                    'Mission.Constraints.EXCESS_FUEL_MASS_CAPACITY for details.'
                )

        self.add_subsystem(
            'block_fuel_comp',
            om.ExecComp(
                'block_fuel = mission_fuel_burned + fuel_burned_taxi_in',
                block_fuel={'units': 'lbm'},
                mission_fuel_burned={'units': 'lbm'},
                fuel_burned_taxi_in={'units': 'lbm'},
            ),
            promotes_inputs=[
                ('mission_fuel_burned', Mission.FUEL_MASS),
                ('fuel_burned_taxi_in', Mission.Taxi.FUEL_MASS_TAXI_IN),
            ],
            promotes_outputs=[('block_fuel', Mission.BLOCK_FUEL_MASS)],
        )
