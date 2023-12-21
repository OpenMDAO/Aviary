import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.examples.external_subsystems.battery.battery_variables import Aircraft, Mission
from aviary.examples.external_subsystems.battery.battery_variable_meta_data import ExtendedMetaData


class BatteryPreMission(om.ExplicitComponent):
    """
    Calculates battery pack mass
    """

    def setup(self):

        # Inputs
        # control from optimizer
        add_aviary_input(self, Aircraft.Battery.ENERGY_REQUIRED,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.EFFICIENCY, meta_data=ExtendedMetaData)

        # output from zappy
        add_aviary_input(self, Aircraft.Battery.CURRENT_MAX, meta_data=ExtendedMetaData)

        # params (fixed)
        add_aviary_input(self, Aircraft.Battery.Cell.HEAT_CAPACITY,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Case.HEAT_CAPACITY,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Cell.MASS, meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Cell.VOLTAGE_LOW,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Cell.DISCHARGE_RATE,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Cell.ENERGY_CAPACITY_MAX,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Case.WEIGHT_FRAC,
                         meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.VOLTAGE, meta_data=ExtendedMetaData)
        add_aviary_input(self, Aircraft.Battery.Cell.VOLUME, meta_data=ExtendedMetaData)

        add_aviary_output(self, Aircraft.Battery.N_SERIES,
                          val=0.0, meta_data=ExtendedMetaData)
        add_aviary_output(self, Aircraft.Battery.N_PARALLEL,
                          val=0.0, meta_data=ExtendedMetaData)

        add_aviary_output(self, Aircraft.Battery.MASS,
                          val=0.0, meta_data=ExtendedMetaData)
        add_aviary_output(self, Aircraft.Battery.HEAT_CAPACITY,
                          val=0.0, meta_data=ExtendedMetaData)
        # unconnected output used for test checking
        add_aviary_output(self, Aircraft.Battery.VOLUME,
                          val=0.0, meta_data=ExtendedMetaData)

        self.declare_partials(Aircraft.Battery.N_SERIES, [
                              Aircraft.Battery.VOLTAGE, Aircraft.Battery.Cell.VOLTAGE_LOW])
        self.declare_partials(Aircraft.Battery.N_PARALLEL, [
                              Aircraft.Battery.CURRENT_MAX, Aircraft.Battery.Cell.DISCHARGE_RATE])
        self.declare_partials(
            Aircraft.Battery.MASS,
            [
                Aircraft.Battery.VOLTAGE,
                Aircraft.Battery.Cell.VOLTAGE_LOW,
                Aircraft.Battery.CURRENT_MAX,
                Aircraft.Battery.Cell.DISCHARGE_RATE,
                Aircraft.Battery.Cell.MASS,
                Aircraft.Battery.Case.WEIGHT_FRAC,
            ],
        )

        self.declare_partials(Aircraft.Battery.HEAT_CAPACITY, [
                              Aircraft.Battery.Case.HEAT_CAPACITY, Aircraft.Battery.Cell.HEAT_CAPACITY, Aircraft.Battery.Case.WEIGHT_FRAC])
        self.declare_partials(
            Aircraft.Battery.VOLUME,
            [
                Aircraft.Battery.VOLTAGE,
                Aircraft.Battery.Cell.VOLTAGE_LOW,
                Aircraft.Battery.CURRENT_MAX,
                Aircraft.Battery.Cell.DISCHARGE_RATE,
                Aircraft.Battery.Cell.VOLUME,
            ],
        )

    def compute(self, inputs, outputs):

        bus_v = inputs[Aircraft.Battery.VOLTAGE]
        c_l_v = inputs[Aircraft.Battery.Cell.VOLTAGE_LOW]
        max_amps = inputs[Aircraft.Battery.CURRENT_MAX]

        outputs[Aircraft.Battery.N_SERIES] = bus_v / c_l_v
        outputs[Aircraft.Battery.N_PARALLEL] = max_amps / \
            inputs[Aircraft.Battery.Cell.DISCHARGE_RATE]
        outputs[Aircraft.Battery.MASS] = (
            outputs[Aircraft.Battery.N_SERIES]
            * outputs[Aircraft.Battery.N_PARALLEL]
            * inputs[Aircraft.Battery.Cell.MASS]
            * inputs[Aircraft.Battery.Case.WEIGHT_FRAC]
        )
        outputs[Aircraft.Battery.HEAT_CAPACITY] = (
            inputs[Aircraft.Battery.Cell.HEAT_CAPACITY] + inputs[Aircraft.Battery.Case.HEAT_CAPACITY] *
            (1.0 - inputs[Aircraft.Battery.Case.WEIGHT_FRAC])
        ) / inputs[Aircraft.Battery.Case.WEIGHT_FRAC]
        outputs[Aircraft.Battery.VOLUME] = (
            outputs[Aircraft.Battery.N_SERIES] *
            outputs[Aircraft.Battery.N_PARALLEL] * inputs[Aircraft.Battery.Cell.VOLUME]
        )

    def compute_partials(self, inputs, partials):

        bus_v = inputs[Aircraft.Battery.VOLTAGE]
        c_l_v = inputs[Aircraft.Battery.Cell.VOLTAGE_LOW]
        max_amps = inputs[Aircraft.Battery.CURRENT_MAX]
        I_rate = inputs[Aircraft.Battery.Cell.DISCHARGE_RATE]
        m_cell = inputs[Aircraft.Battery.Cell.MASS]
        wf_case = inputs[Aircraft.Battery.Case.WEIGHT_FRAC]
        n_s = bus_v / c_l_v
        n_p = max_amps / inputs[Aircraft.Battery.Cell.DISCHARGE_RATE]
        cp_case = inputs[Aircraft.Battery.Case.HEAT_CAPACITY]
        cp_cell = inputs[Aircraft.Battery.Cell.HEAT_CAPACITY]
        vc = inputs[Aircraft.Battery.Cell.VOLUME]

        # n_{series}
        partials[Aircraft.Battery.N_SERIES, Aircraft.Battery.VOLTAGE] = 1.0 / c_l_v
        partials[Aircraft.Battery.N_SERIES,
                 Aircraft.Battery.Cell.VOLTAGE_LOW] = -bus_v / (c_l_v ** 2)
        # n_{parallel}
        partials[Aircraft.Battery.N_PARALLEL,
                 Aircraft.Battery.CURRENT_MAX] = 1.0 / I_rate
        partials[Aircraft.Battery.N_PARALLEL,
                 Aircraft.Battery.Cell.DISCHARGE_RATE] = -max_amps / (I_rate ** 2)
        # mass_{battery}
        partials[Aircraft.Battery.MASS, Aircraft.Battery.VOLTAGE] = (
            partials[Aircraft.Battery.N_SERIES,
                     Aircraft.Battery.VOLTAGE] * n_p * m_cell * wf_case
        )
        partials[Aircraft.Battery.MASS, Aircraft.Battery.Cell.VOLTAGE_LOW] = (
            partials[Aircraft.Battery.N_SERIES,
                     Aircraft.Battery.Cell.VOLTAGE_LOW] * n_p * m_cell * wf_case
        )
        partials[Aircraft.Battery.MASS, Aircraft.Battery.CURRENT_MAX] = (
            partials[Aircraft.Battery.N_PARALLEL,
                     Aircraft.Battery.CURRENT_MAX] * n_s * m_cell * wf_case
        )
        partials[Aircraft.Battery.MASS, Aircraft.Battery.Cell.DISCHARGE_RATE] = (
            partials[Aircraft.Battery.N_PARALLEL,
                     Aircraft.Battery.Cell.DISCHARGE_RATE] * n_s * m_cell * wf_case
        )
        partials[Aircraft.Battery.MASS, Aircraft.Battery.Cell.MASS] = n_s * n_p * wf_case
        partials[Aircraft.Battery.MASS,
                 Aircraft.Battery.Case.WEIGHT_FRAC] = n_s * n_p * m_cell
        # C_{p,batt}
        partials[Aircraft.Battery.HEAT_CAPACITY,
                 Aircraft.Battery.Cell.HEAT_CAPACITY] = 1.0 / wf_case
        partials[Aircraft.Battery.HEAT_CAPACITY,
                 Aircraft.Battery.Case.HEAT_CAPACITY] = (1.0 - wf_case) / wf_case
        partials[Aircraft.Battery.HEAT_CAPACITY, Aircraft.Battery.Case.WEIGHT_FRAC] = (
            -(cp_cell + cp_case) / wf_case ** 2
        )
        # volume_{pack}
        partials[Aircraft.Battery.VOLUME, Aircraft.Battery.VOLTAGE] = (
            partials[Aircraft.Battery.N_SERIES, Aircraft.Battery.VOLTAGE] * n_p * vc
        )
        partials[Aircraft.Battery.VOLUME, Aircraft.Battery.Cell.VOLTAGE_LOW] = (
            partials[Aircraft.Battery.N_SERIES,
                     Aircraft.Battery.Cell.VOLTAGE_LOW] * n_p * vc
        )
        partials[Aircraft.Battery.VOLUME, Aircraft.Battery.CURRENT_MAX] = (
            partials[Aircraft.Battery.N_PARALLEL,
                     Aircraft.Battery.CURRENT_MAX] * n_s * vc
        )
        partials[Aircraft.Battery.VOLUME, Aircraft.Battery.Cell.DISCHARGE_RATE] = (
            partials[Aircraft.Battery.N_PARALLEL,
                     Aircraft.Battery.Cell.DISCHARGE_RATE] * n_s * vc
        )
        partials[Aircraft.Battery.VOLUME, Aircraft.Battery.Cell.VOLUME] = n_s * n_p
