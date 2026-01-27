import openmdao.api as om

from aviary.variable_info.enums import GASPEngineType, Verbosity
from aviary.variable_info.variables import Aircraft, Settings
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output


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
        add_aviary_option(self, Aircraft.Propulsion.TOTAL_NUM_ENGINES)
        add_aviary_option(self, Aircraft.Engine.TYPE)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        add_aviary_input(self, Aircraft.Engine.SCALED_SLS_THRUST)

        add_aviary_output(self, Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, units='lbm')

    def setup_partials(self):
        self.declare_partials('*', '*')

    def compute(self, inputs, outputs):
        verbosity = self.options[Settings.VERBOSITY]
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]
        Fn_SLS = inputs[Aircraft.Engine.SCALED_SLS_THRUST]

        if engine_type is GASPEngineType.TURBOJET:
            oil_per_eng_wt = 0.0054 * Fn_SLS + 12.0
        elif engine_type is GASPEngineType.TURBOSHAFT or engine_type is GASPEngineType.TURBOPROP:
            oil_per_eng_wt = 0.0214 * Fn_SLS + 14
        else:
            # Other engine types are currently not supported in Aviary
            if verbosity > Verbosity.BRIEF:
                print('This engine_type is not curretly supported in Aviary.')
            oil_per_eng_wt = 0

        outputs[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS] = num_engines * oil_per_eng_wt

    def compute_partials(self, J):
        engine_type = self.options[Aircraft.Engine.TYPE][0]
        num_engines = self.options[Aircraft.Propulsion.TOTAL_NUM_ENGINES]

        if engine_type is GASPEngineType.TURBOJET:
            doil_per_eng_wt_dFn_SLS = 0.0054
        elif engine_type is GASPEngineType.TURBOSHAFT or engine_type is GASPEngineType.TURBOPROP:
            doil_per_eng_wt_dFn_SLS = 0.0124
        # else:
        #     doil_per_eng_wt_dFn_SLS = 0.062
        else:
            # Other engine types are currently not supported in Aviary
            doil_per_eng_wt_dFn_SLS = 0.0

        J[Aircraft.Propulsion.TOTAL_ENGINE_OIL_MASS, Aircraft.Engine.SCALED_SLS_THRUST] = (
            doil_per_eng_wt_dFn_SLS * num_engines
        )
