"""
Define subsystem builder for Aviary core mass.

Classes
-------
MassBuilder : the interface for a mass subsystem builder.

CoreMassBuilder : the interface for Aviary's core mass subsystem builder
"""

import numpy as np

from openmdao.utils.units import convert_units

from aviary.interface.utils import find_variable_in_problem
from aviary.subsystems.mass.flops_based.mass_premission import MassPremission as MassPremissionFLOPS
from aviary.subsystems.mass.gasp_based.mass_premission import MassPremission as MassPremissionGASP
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.enums import LegacyCode, ProblemType
from aviary.variable_info.variables import Aircraft, Mission

GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS


class MassBuilder(SubsystemBuilder):
    """
    Base mass builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None):
        Initializes the MassBuilder object with a given name.
    """

    _default_name = 'mass'


class CoreMassBuilder(MassBuilder):
    """Core mass estimation subsystem builder."""

    def __init__(self, name=None, meta_data=None, code_origin=None):
        if code_origin not in (FLOPS, GASP):
            raise ValueError('Code origin is not one of the following: (FLOPS, GASP)')

        self.code_origin = code_origin

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, subsystem_options=None):
        code_origin = self.code_origin
        try:
            method = subsystem_options['method']
        except KeyError:
            method = None
        mass_group = None

        if method != 'external':
            if code_origin is GASP:
                mass_group = MassPremissionGASP()

            elif code_origin is FLOPS:
                mass_group = MassPremissionFLOPS()

        return mass_group

    def build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options):
        # by default there is no mass mission, but call super for safety/future-proofing
        try:
            method = subsystem_options['method']
        except KeyError:
            method = None
        mass_group = None

        if method != 'external':
            mass_group = super().build_mission(
                num_nodes, aviary_inputs, user_options, subsystem_options
            )

        return mass_group

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core mass.

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        filename = self.name + '.md'
        filepath = reports_folder / filename

        # NOTE Workaround due to another workaround in reports.py
        try:
            model = prob.model
        except AttributeError:
            model = prob

        num_engines = model.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)
        engine_models = model.engine_models

        # "double-size" (8-character) tabs were found to greatly improve readability
        tab = '&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'

        with open(filepath, mode='w') as f:
            method = self.code_origin.value + '-derived relations'
            f.write(f'# Mass estimation: {method}')

            f.write('\n| Name | Value | Units |\n')
            f.write('|:-|:-|:-|\n')
            val, units = find_variable_in_problem(Aircraft.Wing.MASS, prob, self.meta_data)
            f.write(f'|Wing|{val}|{units}|\n')

            # EMPENNAGE MASS GROUP #
            val, units = find_variable_in_problem(
                Aircraft.Design.EMPENNAGE_MASS, prob, self.meta_data
            )
            f.write(f'|Empennage Group|{val}|{units}|\n')

            # canard reporting optional
            val, units = find_variable_in_problem(Aircraft.Canard.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Canard|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.HorizontalTail.MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Horizontal Tail|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.VerticalTail.MASS, prob, self.meta_data)
            f.write(f'|{tab}Vertical Tail|{val}|{units}|\n')

            # fin reporting optional
            val, units = find_variable_in_problem(Aircraft.Fins.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Fins|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Fuselage.MASS, prob, self.meta_data)
            f.write(f'|Fuselage|{val}|{units}|\n')

            # LANDING GEAR GROUP #
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.TOTAL_MASS, prob, self.meta_data
            )
            f.write(f'|Landing Gear Group|{val}|{units}|\n')

            # main gear reporting optional
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.MAIN_GEAR_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Main Gear|{val}|{units}|\n')

            # nose gear reporting optional
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.NOSE_GEAR_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Nose Gear|{val}|{units}|\n')

            # OTHER STRUCTURES (NOT IN GROUP) #
            val, units = find_variable_in_problem(Aircraft.Nacelle.MASS, prob, self.meta_data)
            f.write(f'|Nacelles|{np.dot(val, num_engines)}||\n')
            for i, engine in enumerate(engine_models):
                if isinstance(val, (np.ndarray, list, tuple)):
                    val = val[i]
                f.write(f'|{tab}{engine.name}|{val} ({val * num_engines[i]} total)|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Nacelle.MASS, prob, self.meta_data)
            if val == 0.0:
                val = [val]

            # paint reporting optional
            val, units = find_variable_in_problem(Aircraft.Paint.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|Paint|{val}|{units}|\n')

            # additional structural mass reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Design.STRUCTURAL_MASS_INCREMENT, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Additional Structural Mass|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.STRUCTURE_MASS, prob, self.meta_data
            )
            f.write(f'|**Structure Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            # PROPULSION GROUP #
            val, units = find_variable_in_problem(Aircraft.Propulsion.MASS, prob, self.meta_data)
            f.write(f'|Propulsion Group|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_ENGINE_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Engines|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Engine.MASS, prob, self.meta_data)
            for i, engine in enumerate(engine_models):
                if isinstance(val, (np.ndarray, list, tuple)):
                    val = val[i]
                f.write(f'|{tab}{tab}{engine.name}|{val} ({val * num_engines[i]} total)|{units}|\n')

            # thrust reverser reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Thrust Reversers|{val}|{units}|\n')

            # fuel system reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Fuel.FUEL_SYSTEM_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Fuel System|{val}|{units}|\n')

            # misc propulsion mass reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_MISC_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Miscellaneous|{val}|{units}|\n')

            # battery reporting optional
            val, units = find_variable_in_problem(Aircraft.Battery.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Battery|{val}|{units}|\n')

            # SYSTEMS AND EQUIPMENT GROUP #
            val, units = find_variable_in_problem(
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, prob, self.meta_data
            )
            f.write(f'|Systems and Equipment|{val}|{units}|\n')

            # all systems and equipment group items reporting optional
            val, units = find_variable_in_problem(Aircraft.Controls.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Flight Controls|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.APU.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Auxiliary Power|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Instruments.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Instruments|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Hydraulics.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Hydraulics|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Electrical.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Electrical|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Avionics.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Avionics|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Furnishings.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Furnishings|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.AirConditioning.MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Environmental Control|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.APU.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Auxiliary Power|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.AntiIcing.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Anti-Icing|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.OxygenSystem.MASS, prob, self.meta_data)
            if val != 'Not Found in Model':
                f.write(f'|{tab}Oxygen System|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, prob, self.meta_data
            )
            # TODO find individual mass names that got added by each subsystem and report?
            f.write(f'|External Subsystems|{val}|{units}|\n')

            # empty mass margin reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Design.EMPTY_MASS_MARGIN, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|Empty Mass Margin|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Design.EMPTY_MASS, prob, self.meta_data)
            f.write(f'|**Empty Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            # OPERATING ITEMS GROUP #
            val, units = find_variable_in_problem(
                Mission.OPERATING_ITEMS_MASS, prob, self.meta_data
            )
            f.write(f'|Operating Items|{val}|{units}|\n')

            # crew mass reporting optional
            val1, units1 = find_variable_in_problem(
                Aircraft.CrewPayload.FLIGHT_CREW_MASS, prob, self.meta_data
            )
            val2, units2 = find_variable_in_problem(
                Aircraft.CrewPayload.CABIN_CREW_MASS, prob, self.meta_data
            )
            if val1 != 'Not Found in Model' or val2 != 'Not Found in Model':
                crew_sum = 0
                units = units1
                if not isinstance(val1, str):
                    crew_sum += val1
                if not isinstance(val2, str):
                    # need to make sure summing masses of the same units
                    if units1 != units2:
                        crew_sum += convert_units(val2, units2, units1)
                    else:
                        crew_sum += val2
                f.write(f'|{tab}Total Crew|{crew_sum}|{units}|\n')
                f.write(f'|{tab}{tab}Flight Crew|{val1}|{units1}|\n')
                f.write(f'|{tab}{tab}Cabin Crew|{val2}|{units2}|\n')

            # passenger service reporting optional
            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Passenger Service|{val}|{units}|\n')

            # cargo container reporting optional
            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.CARGO_CONTAINER_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Cargo Containers|{val}|{units}|\n')

            # unusable fuel mass reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Fuel.UNUSABLE_FUEL_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Unusable Fuel|{val}|{units}|\n')

            # engine oil reporting optional
            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, prob, self.meta_data
            )
            if val != 'Not Found in Model':
                f.write(f'|{tab}Oil|{val}|{units}|\n')

            val, units = find_variable_in_problem(Mission.OPERATING_MASS, prob, self.meta_data)
            f.write(f'|**Operating Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            # PAYLOAD MASS GROUP #
            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, prob, self.meta_data
            )
            f.write(f'|Payload|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Passengers and Baggage|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.CARGO_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Cargo|{val}|{units}|\n')

            val, units = find_variable_in_problem(Mission.ZERO_FUEL_MASS, prob, self.meta_data)
            f.write(f'|**Zero Fuel Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            # FUEL GROUP #
            val1, units1 = find_variable_in_problem(Mission.FUEL_MASS, prob, self.meta_data)
            val2, units2 = find_variable_in_problem(Mission.RESERVE_FUEL_MASS, prob, self.meta_data)
            if val1 != 'Not Found in Model' or val2 != 'Not Found in Model':
                fuel_sum = 0
                units = units1
                if not isinstance(val1, str):
                    fuel_sum += val1
                if not isinstance(val2, str):
                    # need to make sure summing masses of the same units
                    if units1 != units2:
                        fuel_sum += convert_units(val2, units2, units1)
                    else:
                        fuel_sum += val2
                f.write(f'|Total Fuel|{fuel_sum}|{units}|\n')
                f.write(f'|{tab}Mission Fuel|{val1}|{units1}|\n')
                f.write(f'|{tab}Reserve Fuel|{val2}|{units2}|\n')

            val, units = find_variable_in_problem(Mission.GROSS_MASS, prob, self.meta_data)
            f.write(f'|**Gross Mass**|**{val}**|**{units}**|\n')
