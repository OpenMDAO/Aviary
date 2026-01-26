"""
Define subsystem builder for Aviary core mass.

Classes
-------
MassBuilder : the interface for a mass subsystem builder.

CoreMassBuilder : the interface for Aviary's core mass subsystem builder
"""

import numpy as np

from aviary.interface.utils import find_variable_in_problem
from aviary.subsystems.mass.flops_based.mass_premission import MassPremission as MassPremissionFLOPS
from aviary.subsystems.mass.gasp_based.mass_premission import MassPremission as MassPremissionGASP
from aviary.subsystems.subsystem_builder import SubsystemBuilder
from aviary.variable_info.enums import LegacyCode, ProblemType
from aviary.variable_info.variables import Aircraft, Mission

GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS

_default_name = 'mass'


class MassBuilder(SubsystemBuilder):
    """Base mass builder."""

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)


class CoreMassBuilder(MassBuilder):
    """Core mass subsystem builder."""

    def __init__(self, name=None, meta_data=None, code_origin=None):
        if name is None:
            name = 'mass'

        if code_origin not in (FLOPS, GASP):
            raise ValueError('Code origin is not one of the following: (FLOPS, GASP)')

        self.code_origin = code_origin

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        code_origin = self.code_origin
        try:
            method = kwargs['method']
        except KeyError:
            method = None
        mass_group = None

        if method != 'external':
            if code_origin is GASP:
                mass_group = MassPremissionGASP()

            elif code_origin is FLOPS:
                mass_group = MassPremissionFLOPS()

        return mass_group

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        # by default there is no mass mission, but call super for safety/future-proofing
        try:
            method = kwargs['method']
        except KeyError:
            method = None
        mass_group = None

        if method != 'external':
            mass_group = super().build_mission(num_nodes, aviary_inputs)

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

        # NOTE currently only taking engine information from the first mission
        if prob.problem_type is ProblemType.MULTI_MISSION:
            model = next(iter(prob.aviary_groups_dict.values()))
        else:
            model = prob.model

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
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.EMPENNAGE_MASS, prob, self.meta_data
            )
            f.write(f'|Empennage Group|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Canard.MASS, prob, self.meta_data)
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|{tab}Canard|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.HorizontalTail.MASS, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|{tab}Horizontal Tail|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.VerticalTail.MASS, prob, self.meta_data)
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|{tab}Vertical Tail|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Fins.MASS, prob, self.meta_data)
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|{tab}Fins|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Fuselage.MASS, prob, self.meta_data)
            f.write(f'|Fuselage|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.LandingGear.TOTAL_MASS, prob, self.meta_data
            )
            f.write(f'|Landing Gear Group|{val}|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.MAIN_GEAR_MASS, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|{tab}Main Gear|{val}|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.NOSE_GEAR_MASS, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|{tab}Nose Gear|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Nacelle.MASS, prob, self.meta_data)
            if val == 0.0:
                val = [val]
            f.write(f'|Nacelles|{np.dot(val, num_engines)}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.STRUCTURE_MASS, prob, self.meta_data
            )
            f.write(f'|**Structure Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

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

            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Thrust Reversers|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Fuel.FUEL_SYSTEM_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Fuel System|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_MISC_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Miscellaneous|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Battery.MASS, prob, self.meta_data)
            if val != 0.0 and val != 'Not Found in Model':
                f.write(f'|{tab}Battery|{val}|{units}|\n')
            f.write('|||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, prob, self.meta_data
            )
            f.write(f'|Systems and Equipment|{val}|{units}|\n')
            val, units = find_variable_in_problem(Aircraft.Controls.MASS, prob, self.meta_data)
            f.write(f'|{tab}Flight Controls|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.APU.MASS, prob, self.meta_data)
            f.write(f'|{tab}Auxiliary Power|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Instruments.MASS, prob, self.meta_data)
            f.write(f'|{tab}Instruments|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Hydraulics.MASS, prob, self.meta_data)
            f.write(f'|{tab}Hydraulics|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Electrical.MASS, prob, self.meta_data)
            f.write(f'|{tab}Electrical|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Avionics.MASS, prob, self.meta_data)
            f.write(f'|{tab}Avionics|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Furnishings.MASS, prob, self.meta_data)
            f.write(f'|{tab}Furnishings|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.AirConditioning.MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Environmental Control|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.APU.MASS, prob, self.meta_data)
            f.write(f'|{tab}Auxiliary Power|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.AntiIcing.MASS, prob, self.meta_data)
            f.write(f'|{tab}Anti-Icing|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, prob, self.meta_data
            )
            f.write(f'|External Subsystems|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.EMPTY_MASS_MARGIN, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|Empty Mass Margin|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Design.EMPTY_MASS, prob, self.meta_data)
            f.write(f'|**Empty Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Mission.Summary.USEFUL_LOAD, prob, self.meta_data)
            f.write(f'|Useful Load|{val}|{units}|\n')

            val1, units = find_variable_in_problem(
                Aircraft.CrewPayload.CABIN_CREW_MASS, prob, self.meta_data
            )
            val2, units = find_variable_in_problem(
                Aircraft.CrewPayload.FLIGHT_CREW_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Crew|{val1 + val2}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Passenger Service|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.CARGO_CONTAINER_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Cargo Containers|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Fuel.UNUSABLE_FUEL_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Unusable Fuel|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Oil|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Mission.Summary.OPERATING_MASS, prob, self.meta_data
            )
            f.write(f'|**Operating Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.TOTAL_PAYLOAD_MASS, prob, self.meta_data
            )
            f.write(f'|Payload|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Passengers|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.CARGO_MASS, prob, self.meta_data
            )
            f.write(f'|{tab}Cargo|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Mission.Summary.ZERO_FUEL_MASS, prob, self.meta_data
            )
            f.write(f'|**Zero Fuel Mass**|**{val}**|**{units}**|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Mission.Summary.FUEL_MASS, prob, self.meta_data)
            f.write(f'|Fuel|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Mission.Design.GROSS_MASS, prob, self.meta_data)
            f.write(f'|**Gross Mass**|**{val}**|**{units}**|\n')
