import numpy as np
import openmdao.api as om

from aviary.constants import GRAV_ENGLISH_LBM
from aviary.variable_info.enums import GASPEngineType, Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings


class EngineOilMass(om.ExplicitComponent):
    """
    Calculates the mass of engine oil using the transport/general aviation method.
    The methodology is based on the GASP weight equations, modified to output mass
    instead of weight.

    Assumptions
    -----------
    Calculates total, propulsion-system level mass of all engine oil

    All engines assumed to use engine oil whose mass follows this equation
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_engine_types = len(self.options[Aircraft.Engine.NUM_ENGINES])
        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST, shape=num_engine_types)

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        engine_type = self.options[Aircraft.Engine.TYPE]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_types = len(num_engines)
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        oil_per_eng_wt = np.zeros(num_engine_types, dtype=Fn_SLS.dtype)

        for i, etype in enumerate(engine_type):
            if etype is GASPEngineType.TURBOJET:
                oil_per_eng_wt[i] = 0.0054 * Fn_SLS[i] + 12.0
            elif etype is GASPEngineType.TURBOSHAFT or etype is GASPEngineType.TURBOPROP:
                oil_per_eng_wt[i] = 0.0214 * Fn_SLS[i] + 14
            else:
                # Other engine types are currently not supported in Aviary
                if verbosity > Verbosity.BRIEF:
                    print(
                        f"Engine type {etype} is not supported by Aviary's implementation of GASP mass methodology."
                    )
                oil_per_eng_wt[i] = 0

        outputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS] = (
            np.dot(oil_per_eng_wt, num_engines) / GRAV_ENGLISH_LBM
        )

    def compute_partials(self, inputs, J):
        engine_type = self.options[Aircraft.Engine.TYPE]
        num_engines = self.options[Aircraft.Engine.NUM_ENGINES]
        num_engine_types = len(num_engines)

        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        doil_per_eng_wt_dFn_SLS = np.zeros(num_engine_types, dtype=Fn_SLS.dtype)

        for i, etype in enumerate(engine_type):
            if etype is GASPEngineType.TURBOJET:
                doil_per_eng_wt_dFn_SLS[i] = 0.0054
            elif etype is GASPEngineType.TURBOSHAFT or etype is GASPEngineType.TURBOPROP:
                doil_per_eng_wt_dFn_SLS[i] = 0.0214
            # else:
            #     doil_per_eng_wt_dFn_SLS = 0.062
            else:
                # Other engine types are currently not supported in Aviary
                doil_per_eng_wt_dFn_SLS[i] = 0.0

        J[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            doil_per_eng_wt_dFn_SLS * num_engines / GRAV_ENGLISH_LBM
        )
