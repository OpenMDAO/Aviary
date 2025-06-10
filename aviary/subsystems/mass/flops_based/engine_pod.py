import numpy as np
import openmdao.api as om

from aviary.subsystems.mass.flops_based.distributed_prop import nacelle_count_factor
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft


class EnginePodMass(om.ExplicitComponent):
    """
    Calculates the mass of a single engine pod for each unique engine type.

    ASSUMPTIONS:
     - System-level masses are estimated per engine by normalizing with ratio of total
       engine model set's thrust to total aircraft thrust
     - Engine mount location (wing vs. fueselage) has no impact on pod mass
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Electrical.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Fuel.FUEL_SYSTEM_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Hydraulics.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Instruments.MASS, units='lbm')
        add_aviary_input(self, Aircraft.Nacelle.MASS, shape=num_engine_type, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS, units='lbm')
        add_aviary_input(self, Aircraft.Engine.MASS, shape=num_engine_type, units='lbm')
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_STARTER_MASS, units='lbm')
        add_aviary_input(
            self, Aircraft.Engine.THRUST_REVERSERS_MASS, shape=num_engine_type, units='lbm'
        )
        add_aviary_input(
            self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_type, units='lbf'
        )
        add_aviary_input(self, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST, units='lbf')

        add_aviary_output(self, Aircraft.Engine.POD_MASS, shape=num_engine_type, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

        # derivatives w.r.t vectorized engine inputs have known sparsity pattern
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Engine.POD_MASS,
            Aircraft.Engine.THRUST_REVERSERS_MASS,
            rows=shape,
            cols=shape,
            val=1.0,
        )
        self.declare_partials(
            Aircraft.Engine.POD_MASS, Aircraft.Engine.MASS, rows=shape, cols=shape, val=1.0
        )
        self.declare_partials(
            Aircraft.Engine.POD_MASS, Aircraft.Nacelle.MASS, rows=shape, cols=shape, val=1.0
        )
        self.declare_partials(
            Aircraft.Engine.POD_MASS,
            Aircraft.Engine.SCALED_SLS_THRUST,
            rows=shape,
            cols=shape,
            val=1.0,
        )

    def compute(self, inputs, outputs, discrete_inputs=None, discrete_outputs=None):
        # BUG this methodology completely ignores miscellaneous mass. There is a discrepancy between this calculation
        #     and miscellaneous mass. Engine control, starter, and additional mass have a scaler applied to them, and
        #     if their calculated values are used directly this scaler is skipped
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]
        nacelle_count = nacelle_count_factor(num_eng)

        eng_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        total_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]

        # NOTE if this value gets needed elsewhere, calculate in pre-mission propulsion
        #      and pass to this component as input
        engine_thrust_ratio = (eng_thrust * num_eng) / total_thrust

        m_eng = inputs[Aircraft.Engine.MASS]
        m_thr = inputs[Aircraft.Engine.THRUST_REVERSERS_MASS]
        m_start = inputs[Aircraft.Propulsion.TOTAL_STARTER_MASS]
        m_ctrl = inputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS]
        m_inst = inputs[Aircraft.Instruments.MASS]
        m_elec = inputs[Aircraft.Electrical.MASS]
        m_hyd = inputs[Aircraft.Hydraulics.MASS]
        m_fsys = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS]
        m_nac = inputs[Aircraft.Nacelle.MASS]

        # calculate nacelle content mass for engine set
        nacelle_content_mass = (
            (num_eng * m_eng)
            + m_thr
            + engine_thrust_ratio
            * (m_start + 0.25 * (m_ctrl + m_fsys) + 0.13 * (m_elec + m_hyd) + 0.11 * m_inst)
        )

        pod_mass = np.array([])

        # calculate engine pod mass for single engine of each type
        for i in range(len(num_eng)):
            pod_mass = np.append(
                pod_mass,
                nacelle_content_mass[i] / max(1, num_eng[i]) + m_nac[i] / max(1, nacelle_count[i]),
            )

        outputs[Aircraft.Engine.POD_MASS] = pod_mass

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        num_eng = self.options[Aircraft.Engine.NUM_ENGINES]
        count_factor = nacelle_count_factor(num_eng)

        m_start = inputs[Aircraft.Propulsion.TOTAL_STARTER_MASS]
        m_ctrl = inputs[Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS]
        m_inst = inputs[Aircraft.Instruments.MASS]
        m_elec = inputs[Aircraft.Electrical.MASS]
        m_hyd = inputs[Aircraft.Hydraulics.MASS]
        m_fsys = inputs[Aircraft.Fuel.FUEL_SYSTEM_MASS]

        eng_thrust = inputs[Aircraft.Engine.SCALED_SLS_THRUST]
        total_thrust = inputs[Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST]
        # TODO this value may be needed elsewhere, if so, calculate in static
        # propulsion and pass to this component as input
        ratio = eng_thrust * num_eng / total_thrust

        fact1 = np.array([1.0 / max(1, eng) for eng in num_eng])
        fact2 = np.array([1.0 / max(1, fact) for fact in count_factor])

        nac_fact = m_start + 0.25 * (m_ctrl + m_fsys) + 0.13 * (m_elec + m_hyd) + 0.11 * m_inst

        partials[Aircraft.Engine.POD_MASS, Aircraft.Engine.MASS] = 1.0

        partials[Aircraft.Engine.POD_MASS, Aircraft.Engine.THRUST_REVERSERS_MASS] = fact1

        partials[Aircraft.Engine.POD_MASS, Aircraft.Propulsion.TOTAL_STARTER_MASS] = fact1 * ratio

        partials[Aircraft.Engine.POD_MASS, Aircraft.Propulsion.TOTAL_ENGINE_CONTROLS_MASS] = (
            0.25 * fact1 * ratio
        )

        partials[Aircraft.Engine.POD_MASS, Aircraft.Instruments.MASS] = 0.11 * fact1 * ratio

        partials[Aircraft.Engine.POD_MASS, Aircraft.Electrical.MASS] = 0.13 * fact1 * ratio

        partials[Aircraft.Engine.POD_MASS, Aircraft.Hydraulics.MASS] = 0.13 * fact1 * ratio

        partials[Aircraft.Engine.POD_MASS, Aircraft.Fuel.FUEL_SYSTEM_MASS] = 0.25 * fact1 * ratio

        partials[Aircraft.Engine.POD_MASS, Aircraft.Nacelle.MASS] = fact2

        partials[Aircraft.Engine.POD_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            num_eng * nac_fact * fact1
        ) / total_thrust

        partials[Aircraft.Engine.POD_MASS, Aircraft.Propulsion.TOTAL_SCALED_SLS_THRUST] = -(
            num_eng * eng_thrust * nac_fact * fact1
        ) / (total_thrust**2)
