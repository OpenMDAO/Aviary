import numpy as np

import openmdao.api as om

from aviary.subsystems.mass.flops_based.empty_margin import EmptyMassMargin
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft, Mission


class MassSummation(om.Group):
    """
    Group to compute various design masses for FLOPS-based mass:
    Aircraft.Design.STRUCTURE_MASS, Aircraft.Propulsion.MASS, 
    Aircraft.Design.SYSTEMS_EQUIP_MASS, Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE,
    Aircraft.Design.SYSTEMS_EQUIP_MASS, Aircraft.Design.EMPTY_MASS,
    Aircraft.Design.OPERATING_MASS, Aircraft.Design.ZERO_FUEL_MASS,
    Mission.Design.FUEL_MASS.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        aviary_options: AviaryValues = self.options['aviary_options']
        alt_mass = aviary_options.get_val(Aircraft.Design.USE_ALT_MASS)

        self.add_subsystem(
            'structure_mass', StructureMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'propulsion_mass', PropulsionMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        if alt_mass:
            self.add_subsystem(
                'system_equip_mass_base',
                AltSystemsEquipMassBase(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

            self.add_subsystem(
                'system_equip_mass',
                AltSystemsEquipMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        else:
            self.add_subsystem(
                'system_equip_mass', SystemsEquipMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'empty_mass_margin', EmptyMassMargin(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        if alt_mass:
            self.add_subsystem(
                'empty_mass', AltEmptyMass(aviary_options=aviary_options),
                promotes_inputs=['*'], promotes_outputs=['*'])

        else:
            self.add_subsystem('empty_mass', EmptyMass(aviary_options=aviary_options),
                               promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'operating_mass', OperatingMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem(
            'zero_fuel_mass', ZeroFuelMass(aviary_options=aviary_options),
            promotes_inputs=['*'], promotes_outputs=['*'])

        self.add_subsystem('fuel_mass', FuelMass(aviary_options=aviary_options),
                           promotes_inputs=['*'], promotes_outputs=['*'])


class StructureMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))

        add_aviary_input(self, Aircraft.Canard.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Fins.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Fuselage.MASS, val=0.0)
        add_aviary_input(self, Aircraft.HorizontalTail.MASS, val=0.0)
        add_aviary_input(self, Aircraft.LandingGear.MAIN_GEAR_MASS, val=0.0)
        add_aviary_input(self, Aircraft.LandingGear.NOSE_GEAR_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Nacelle.MASS, val=np.zeros(num_engine_type))
        add_aviary_input(self, Aircraft.Paint.MASS, val=0.0)
        add_aviary_input(self, Aircraft.VerticalTail.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Wing.MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.STRUCTURE_MASS, val=0.0)

    def setup_partials(self):
        num_engine_type = len(self.options['aviary_options'].get_val(
            Aircraft.Engine.NUM_ENGINES))

        self.declare_partials(Aircraft.Design.STRUCTURE_MASS, '*', val=1)
        self.declare_partials(Aircraft.Design.STRUCTURE_MASS, Aircraft.Nacelle.MASS,
                              val=np.ones(num_engine_type))

    def compute(self, inputs, outputs):
        canard_mass = inputs[Aircraft.Canard.MASS]
        fin_mass = inputs[Aircraft.Fins.MASS]
        fus_mass = inputs[Aircraft.Fuselage.MASS]
        htail_mass = inputs[Aircraft.HorizontalTail.MASS]
        main_gear_mass = inputs[Aircraft.LandingGear.MAIN_GEAR_MASS]
        nose_gear_mass = inputs[Aircraft.LandingGear.NOSE_GEAR_MASS]
        nac_mass = inputs[Aircraft.Nacelle.MASS]
        paint_mass = inputs[Aircraft.Paint.MASS]
        vtail_mass = inputs[Aircraft.VerticalTail.MASS]
        wing_mass = inputs[Aircraft.Wing.MASS]

        outputs[Aircraft.Design.STRUCTURE_MASS] = \
            wing_mass + htail_mass + vtail_mass + fin_mass + canard_mass + \
            fus_mass + main_gear_mass + nose_gear_mass + np.sum(nac_mass) + paint_mass


class PropulsionMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_MISC_MASS, val=0.0)
        add_aviary_input(
            self, Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Propulsion.MASS, val=0.0)

    def setup_partials(self):
        prop_wrt = [
            Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS,
            Aircraft.Propulsion.TOTAL_MISC_MASS,
            Aircraft.Fuel.FUEL_SYSTEM_MASS,
            Aircraft.Propulsion.TOTAL_ENGINE_MASS]

        self.declare_partials(Aircraft.Propulsion.MASS, prop_wrt, val=1)

    def compute(self, inputs, outputs):
        fuel_sys_mass = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS]
        misc_prop_mass = inputs[Aircraft.Propulsion.TOTAL_MISC_MASS]
        thrust_rev_mass = inputs[Aircraft.Propulsion.TOTAL_THRUST_REVERSERS_MASS]
        total_eng_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_MASS]

        outputs[Aircraft.Propulsion.MASS] = \
            thrust_rev_mass + misc_prop_mass + fuel_sys_mass + total_eng_mass


class SystemsEquipMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS, val=0.0)
        add_aviary_input(self, Aircraft.AntiIcing.MASS, val=0.0)
        add_aviary_input(self, Aircraft.APU.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Avionics.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Electrical.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Furnishings.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Hydraulics.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Instruments.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.SYSTEMS_EQUIP_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.SYSTEMS_EQUIP_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        AC_mass = inputs[Aircraft.AirConditioning.MASS]
        anti_icing_mass = inputs[Aircraft.AntiIcing.MASS]
        APU_mass = inputs[Aircraft.APU.MASS]
        avionics_mass = inputs[Aircraft.Avionics.MASS]
        elec_mass = inputs[Aircraft.Electrical.MASS]
        furnish_mass = inputs[Aircraft.Furnishings.MASS]
        hydraulics_mass = inputs[Aircraft.Hydraulics.MASS]
        instrument_mass = inputs[Aircraft.Instruments.MASS]
        surf_ctrl_mass = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS]
        subsystems_mass = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        outputs[Aircraft.Design.SYSTEMS_EQUIP_MASS] = \
            surf_ctrl_mass + APU_mass + instrument_mass + hydraulics_mass + \
            elec_mass + avionics_mass + furnish_mass + AC_mass + anti_icing_mass + \
            subsystems_mass


class AltSystemsEquipMassBase(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.AirConditioning.MASS, val=0.0)
        add_aviary_input(self, Aircraft.AntiIcing.MASS, val=0.0)
        add_aviary_input(self, Aircraft.APU.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Avionics.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Electrical.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Furnishings.MASS_BASE, val=0.0)
        add_aviary_input(self, Aircraft.Hydraulics.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Instruments.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Wing.SURFACE_CONTROL_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE, '*', val=1)

    def compute(self, inputs, outputs):
        AC_mass = inputs[Aircraft.AirConditioning.MASS]
        anti_icing_mass = inputs[Aircraft.AntiIcing.MASS]
        APU_mass = inputs[Aircraft.APU.MASS]
        avionics_mass = inputs[Aircraft.Avionics.MASS]
        elec_mass = inputs[Aircraft.Electrical.MASS]
        furnish_mass_base = inputs[Aircraft.Furnishings.MASS_BASE]
        hydraulics_mass = inputs[Aircraft.Hydraulics.MASS]
        instrument_mass = inputs[Aircraft.Instruments.MASS]
        surf_ctrl_mass = inputs[Aircraft.Wing.SURFACE_CONTROL_MASS]
        subsystems_mass = inputs[Aircraft.Design.EXTERNAL_SUBSYSTEMS_MASS]

        outputs[Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE] = \
            surf_ctrl_mass + APU_mass + instrument_mass + hydraulics_mass + \
            elec_mass + avionics_mass + furnish_mass_base + AC_mass + anti_icing_mass + \
            subsystems_mass


class AltSystemsEquipMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE, val=0.0)
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.SYSTEMS_EQUIP_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(
            Aircraft.Design.SYSTEMS_EQUIP_MASS,
            [Aircraft.Design.STRUCTURE_MASS, Aircraft.Propulsion.MASS],
            val=0.01
        )

        self.declare_partials(
            Aircraft.Design.SYSTEMS_EQUIP_MASS,
            Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE,
            val=1.01
        )

    def compute(self, inputs, outputs):
        sys_equip_mass_base = inputs[Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE]
        structure_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]

        outputs[Aircraft.Design.SYSTEMS_EQUIP_MASS] = \
            sys_equip_mass_base + 0.01 * \
            (structure_mass + prop_mass + sys_equip_mass_base)


class EmptyMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS_MARGIN, val=0.0)
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.SYSTEMS_EQUIP_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.EMPTY_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.EMPTY_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        structure_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        sys_equip_mass = inputs[Aircraft.Design.SYSTEMS_EQUIP_MASS]
        empty_mass_margin = inputs[Aircraft.Design.EMPTY_MASS_MARGIN]

        outputs[Aircraft.Design.EMPTY_MASS] = \
            structure_mass + prop_mass + sys_equip_mass + empty_mass_margin


class AltEmptyMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS_MARGIN, val=0.0)
        add_aviary_input(self, Aircraft.Design.STRUCTURE_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE, val=0.0)

        add_aviary_output(self, Aircraft.Design.EMPTY_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.EMPTY_MASS,
                              [Aircraft.Design.STRUCTURE_MASS,
                               Aircraft.Propulsion.MASS,
                               Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE],
                              val=1.01)
        self.declare_partials(Aircraft.Design.EMPTY_MASS,
                              Aircraft.Design.EMPTY_MASS_MARGIN,
                              val=1.0)

    def compute(self, inputs, outputs):
        structure_mass = inputs[Aircraft.Design.STRUCTURE_MASS]
        prop_mass = inputs[Aircraft.Propulsion.MASS]
        sys_equip_mass_base = inputs[Aircraft.Design.SYSTEMS_EQUIP_MASS_BASE]
        empty_mass_margin = inputs[Aircraft.Design.EMPTY_MASS_MARGIN]

        outputs[Aircraft.Design.EMPTY_MASS] = \
            1.01 * (structure_mass + prop_mass + sys_equip_mass_base) + empty_mass_margin


class OperatingMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_CONTAINER_MASS, val=0.0)
        add_aviary_input(self, Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS, val=0.0)
        add_aviary_input(self, Aircraft.CrewPayload.FLIGHT_CREW_MASS, val=0.0)
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_SERVICE_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.EMPTY_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Fuel.UNUSABLE_FUEL_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.OPERATING_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.OPERATING_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        cargo_container_mass = inputs[Aircraft.CrewPayload.CARGO_CONTAINER_MASS]
        non_flight_crew_mass = inputs[Aircraft.CrewPayload.NON_FLIGHT_CREW_MASS]
        flight_crew_mass = inputs[Aircraft.CrewPayload.FLIGHT_CREW_MASS]
        empty_mass = inputs[Aircraft.Design.EMPTY_MASS]
        oil_mass = inputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS]
        pass_service_mass = inputs[Aircraft.CrewPayload.PASSENGER_SERVICE_MASS]
        unusable_fuel_mass = inputs[Aircraft.Fuel.UNUSABLE_FUEL_MASS]

        outputs[Aircraft.Design.OPERATING_MASS] = \
            empty_mass + non_flight_crew_mass + flight_crew_mass + \
            unusable_fuel_mass + oil_mass + pass_service_mass + cargo_container_mass


class ZeroFuelMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Aircraft.CrewPayload.PASSENGER_MASS, val=0.0)
        add_aviary_input(self, Aircraft.CrewPayload.BAGGAGE_MASS, val=0.0)
        add_aviary_input(self, Aircraft.CrewPayload.CARGO_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.OPERATING_MASS, val=0.0)

        add_aviary_output(self, Aircraft.Design.ZERO_FUEL_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Aircraft.Design.ZERO_FUEL_MASS, '*', val=1)

    def compute(self, inputs, outputs):
        pass_mass = inputs[Aircraft.CrewPayload.PASSENGER_MASS]
        bag_mass = inputs[Aircraft.CrewPayload.BAGGAGE_MASS]
        cargo_mass = inputs[Aircraft.CrewPayload.CARGO_MASS]
        operating_mass = inputs[Aircraft.Design.OPERATING_MASS]

        outputs[Aircraft.Design.ZERO_FUEL_MASS] = \
            operating_mass + pass_mass + bag_mass + cargo_mass


class FuelMass(om.ExplicitComponent):

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        add_aviary_input(self, Mission.Design.GROSS_MASS, val=0.0)
        add_aviary_input(self, Aircraft.Design.ZERO_FUEL_MASS, val=0.0)

        add_aviary_output(self, Mission.Design.FUEL_MASS, val=0.0)

    def setup_partials(self):
        self.declare_partials(Mission.Design.FUEL_MASS,
                              Mission.Design.GROSS_MASS, val=1)
        self.declare_partials(Mission.Design.FUEL_MASS,
                              Aircraft.Design.ZERO_FUEL_MASS, val=-1)

    def compute(self, inputs, outputs):
        zero_fuel_mass = inputs[Aircraft.Design.ZERO_FUEL_MASS]
        gross_mass = inputs[Mission.Design.GROSS_MASS]

        outputs[Mission.Design.FUEL_MASS] = gross_mass - zero_fuel_mass
