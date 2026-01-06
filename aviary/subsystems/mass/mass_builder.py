"""
Define subsystem builder for Aviary core mass.

Classes
-------
MassBuilderBase : the interface for a mass subsystem builder.

CoreMassBuilder : the interface for Aviary's core mass subsystem builder
"""

import numpy as np

from aviary.interface.utils import find_variable_in_problem, write_markdown_variable_table
from aviary.subsystems.mass.flops_based.mass_premission import MassPremission as MassPremissionFLOPS
from aviary.subsystems.mass.gasp_based.mass_premission import MassPremission as MassPremissionGASP
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft, Mission

GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS

_default_name = 'mass'


class MassBuilderBase(SubsystemBuilderBase):
    """Base mass builder."""

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CoreMassBuilder(MassBuilderBase):
    """Core mass subsystem builder."""

    def __init__(self, name=None, meta_data=None, code_origin=None):
        if name is None:
            name = 'core_mass'

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

        # outputs = [
        #     Aircraft.Wing.MASS,
        #     Aircraft.Design.EMPENNAGE_MASS,
        #     Aircraft.Canard.MASS,
        #     Aircraft.HorizontalTail.MASS,
        #     Aircraft.VerticalTail.MASS,
        #     Aircraft.Fins.MASS,
        #     Aircraft.Fuselage.MASS,
        #     Aircraft.LandingGear.TOTAL_MASS,
        #     Aircraft.LandingGear.MAIN_GEAR_MASS,
        #     Aircraft.LandingGear.NOSE_GEAR_MASS,
        #     Aircraft.Nacelle.MASS,
        #     Aircraft.Design.STRUCTURE_MASS,
        #     Aircraft.Propulsion.MASS,
        #     Aircraft.Engine.MASS,
        #     Aircraft.Engine.CONTROLS_MASS,
        #     Aircraft.Engine.STARTER_MASS,
        #     # Aircraft.Engine.Propeller.MASS
        #     Aircraft.Fuel.FUEL_SYSTEM_MASS,
        #     Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS,
        #     Aircraft.Controls.MASS,
        #     Aircraft.APU.MASS,
        #     Aircraft.Instruments.MASS,
        #     Aircraft.Hydraulics.MASS,
        #     Aircraft.Electrical.MASS,
        #     Aircraft.Avionics.MASS,
        #     Aircraft.Furnishings.MASS,
        #     Aircraft.AirConditioning.MASS,
        #     Aircraft.AntiIcing.MASS,
        #     Aircraft.Design.EMPTY_MASS,
        #     Mission.Summary.USEFUL_LOAD,
        #     Aircraft.CrewPayload.FLIGHT_CREW_MASS,
        #     Aircraft.CrewPayload.CABIN_CREW_MASS,
        #     Aircraft.Fuel.UNUSABLE_FUEL_MASS,
        #     Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
        #     Mission.Summary.OPERATING_MASS,
        #     Aircraft.CrewPayload.PASSENGER_MASS_TOTAL,
        #     Aircraft.CrewPayload.CARGO_MASS,
        #     Aircraft.CrewPayload.WING_CARGO,
        #     Aircraft.CrewPayload.MISC_CARGO,
        #     Mission.Summary.ZERO_FUEL_MASS,
        #     Mission.Summary.FUEL_MASS,
        #     Mission.Summary.GROSS_MASS,
        # ]

        num_engines = prob.model.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)
        engine_models = prob.model.engine_builders

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
                f.write(f'|\tCanard|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.HorizontalTail.MASS, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|\tHorizontal Tail|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.VerticalTail.MASS, prob, self.meta_data)
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|\tVertical Tail|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Fins.MASS, prob, self.meta_data)
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|\tFins|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Fuselage.MASS, prob, self.meta_data)
            f.write(f'|Fuselage|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.LandingGear.TOTAL_MASS, prob, self.meta_data
            )
            f.write(f'|\tLanding Gear Group|{val}|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.MAIN_GEAR_MASS, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|\tMain Gear|{val}|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.LandingGear.NOSE_GEAR_MASS, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|\tNose Gear|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Nacelle.MASS, prob, self.meta_data)
            if val == 0.0:
                val = [val]
            f.write(f'|Nacelles|{np.dot(val, num_engines)}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.STRUCTURE_MASS, prob, self.meta_data
            )
            f.write(f'|Total Structure|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Aircraft.Propulsion.MASS, prob, self.meta_data)
            f.write(f'|Propulsion Group|{val}|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_ENGINE_MASS, prob, self.meta_data
            )
            f.write(f'|\tEngines|{val}|{units}|\n')
            val, units = find_variable_in_problem(Aircraft.Engine.MASS, prob, self.meta_data)
            for i, engine in enumerate(engine_models):
                if isinstance(val, (np.ndarray, list, tuple)):
                    val = val[i]
                f.write(f'|\t\t{engine.name}|{val} ({val * num_engines[i]} total)|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, prob, self.meta_data
            )

            f.write(f'|\tEngine Controls|{val}|{units}|\n')
            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_STARTER_MASS, prob, self.meta_data
            )
            f.write(f'|\tStarting System|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Battery.MASS, prob, self.meta_data)
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|\tBattery|{val}|{units}|\n')
            f.write('|||\n')

            val, units = find_variable_in_problem(Aircraft.Controls.MASS, prob, self.meta_data)
            f.write(f'|Flight Controls|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.APU.MASS, prob, self.meta_data)
            f.write(f'|Auxiliary Power|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Instruments.MASS, prob, self.meta_data)
            f.write(f'|Instruments|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Hydraulics.MASS, prob, self.meta_data)
            f.write(f'|Hydraulics|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Electrical.MASS, prob, self.meta_data)
            f.write(f'|Electrical|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Avionics.MASS, prob, self.meta_data)
            f.write(f'|Avionics|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Furnishings.MASS, prob, self.meta_data)
            f.write(f'|Furnishings|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.AirConditioning.MASS, prob, self.meta_data
            )
            f.write(f'|Environmental Control|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.APU.MASS, prob, self.meta_data)
            f.write(f'|Auxiliary Power|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.AntiIcing.MASS, prob, self.meta_data)
            f.write(f'|Anti-Icing|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.CARGO_CONTAINER_MASS, prob, self.meta_data
            )
            f.write(f'|Load & Handling|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.SYSTEMS_AND_EQUIPMENT_MASS, prob, self.meta_data
            )
            f.write(f'|Total Systems and Equipment|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.Design.EMPTY_MASS_MARGIN, prob, self.meta_data
            )
            if val != 0.0 or val != 'Not Found in Model':
                f.write(f'|Empty Mass Margin|{val}|{units}|\n')

            val, units = find_variable_in_problem(Aircraft.Design.EMPTY_MASS, prob, self.meta_data)
            f.write(f'|Empty Mass|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Mission.Summary.USEFUL_LOAD, prob, self.meta_data)
            f.write(f'|Useful Load|{val}|{units}|\n')

            val1, units = find_variable_in_problem(
                Aircraft.CrewPayload.CABIN_CREW_MASS, prob, self.meta_data
            )
            val2, units = find_variable_in_problem(
                Aircraft.CrewPayload.FLIGHT_CREW_MASS, prob, self.meta_data
            )
            f.write(f'|\tCrew|{val1 + val2}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Fuel.UNUSABLE_FUEL_MASS, prob, self.meta_data
            )
            f.write(f'|\tUnusable Fuel|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, prob, self.meta_data
            )
            f.write(f'|\tOil|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Mission.Summary.OPERATING_MASS, prob, self.meta_data
            )
            f.write(f'|Operating Mass|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS, prob, self.meta_data
            )
            f.write(f'|Passengers|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Aircraft.CrewPayload.CARGO_MASS, prob, self.meta_data
            )
            f.write(f'|Cargo|{val}|{units}|\n')

            val, units = find_variable_in_problem(
                Mission.Summary.ZERO_FUEL_MASS, prob, self.meta_data
            )
            f.write(f'|Zero Fuel Mass|{val}|{units}|\n')
            f.write('||||\n')

            val, units = find_variable_in_problem(Mission.Summary.FUEL_MASS, prob, self.meta_data)
            f.write(f'|Fuel|{val}|{units}|\n')

            val, units = find_variable_in_problem(Mission.Design.GROSS_MASS, prob, self.meta_data)
            f.write(f'|Gross Mass|{val}|{units}|\n')
