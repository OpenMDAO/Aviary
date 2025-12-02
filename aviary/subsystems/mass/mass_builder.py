"""
Define subsystem builder for Aviary core mass.

Classes
-------
MassBuilderBase : the interface for a mass subsystem builder.

CoreMassBuilder : the interface for Aviary's core mass subsystem builder
"""

from aviary.interface.utils import write_markdown_variable_table
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

        # TODO break out table into constituent categories (SAWE standard??)
        outputs = [
            Aircraft.Wing.MASS,
            Aircraft.HorizontalTail.MASS,
            Aircraft.VerticalTail.MASS,
            Aircraft.Fins.MASS,
            Aircraft.Canard.MASS,
            Aircraft.Fuselage.MASS,
            Aircraft.LandingGear.NOSE_GEAR_MASS,
            Aircraft.LandingGear.MAIN_GEAR_MASS,
            Aircraft.Paint.MASS,
            Aircraft.Nacelle.MASS,
            Aircraft.Design.STRUCTURE_MASS,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS,
            Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
            Aircraft.Propulsion.TOTAL_MISC_MASS,
            Aircraft.Fuel.FUEL_SYSTEM_MASS,
            Aircraft.Propulsion.MASS,
            Aircraft.Wing.SURFACE_CONTROL_MASS,
            Aircraft.APU.MASS,
            Aircraft.Instruments.MASS,
            Aircraft.Hydraulics.MASS,
            Aircraft.Electrical.MASS,
            Aircraft.Avionics.MASS,
            Aircraft.Furnishings.MASS,
            Aircraft.AirConditioning.MASS,
            Aircraft.AntiIcing.MASS,
            Aircraft.Design.SYSTEMS_EQUIP_MASS,
            Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS,
            Aircraft.Design.EMPTY_MASS,
            Aircraft.CrewPayload.FLIGHT_CREW_MASS,
            Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS,
            Aircraft.Fuel.UNUSABLE_FUEL_MASS,
            Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS,
            Aircraft.CrewPayload.PASSENGER_SERVICE_MASS,
            Aircraft.CrewPayload.CARGO_CONTAINER_MASS,
            Mission.Summary.OPERATING_MASS,
            Aircraft.CrewPayload.PASSENGER_MASS,
            Aircraft.CrewPayload.PASSENGER_PAYLOAD_MASS,
            Aircraft.CrewPayload.CARGO_MASS,
            Mission.Summary.ZERO_FUEL_MASS,
            Mission.Summary.FUEL_MASS,
            Mission.Summary.TOTAL_FUEL_MASS,
            Mission.Summary.GROSS_MASS,
        ]

        with open(filepath, mode='w') as f:
            method = self.code_origin.value + '-derived relations'
            f.write(f'# Mass estimation: {method}')
            write_markdown_variable_table(f, prob, outputs, self.meta_data)
