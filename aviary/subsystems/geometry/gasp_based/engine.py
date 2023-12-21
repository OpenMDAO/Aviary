import numpy as np
import openmdao.api as om

from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.functions import add_aviary_input, add_aviary_output
from aviary.variable_info.variables import Aircraft


class EngineSize(om.ExplicitComponent):
    """GASP engine geometry calculation."""

    def initialize(self):

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

    def setup(self):
        add_aviary_input(self, Aircraft.Engine.REFERENCE_DIAMETER, 5.8)
        add_aviary_input(self, Aircraft.Engine.SCALE_FACTOR, 1.0)
        add_aviary_input(self, Aircraft.Nacelle.CORE_DIAMETER_RATIO, 1.25)
        add_aviary_input(self, Aircraft.Nacelle.FINENESS, 2)

        add_aviary_output(self, Aircraft.Nacelle.AVG_DIAMETER, val=0.0)
        add_aviary_output(self, Aircraft.Nacelle.AVG_LENGTH, val=0.0)
        add_aviary_output(self, Aircraft.Nacelle.SURFACE_AREA, val=0.0)

    def setup_partials(self):
        innames = [
            Aircraft.Engine.REFERENCE_DIAMETER,
            Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Nacelle.CORE_DIAMETER_RATIO,
            Aircraft.Nacelle.FINENESS,
        ]
        self.declare_partials(Aircraft.Nacelle.AVG_DIAMETER, innames[:-1])
        self.declare_partials(Aircraft.Nacelle.AVG_LENGTH, innames)
        self.declare_partials(Aircraft.Nacelle.SURFACE_AREA, innames)

    def compute(self, inputs, outputs):
        d_ref = inputs[Aircraft.Engine.REFERENCE_DIAMETER]
        scale_fac = inputs[Aircraft.Engine.SCALE_FACTOR]
        d_nac_eng = inputs[Aircraft.Nacelle.CORE_DIAMETER_RATIO]
        ld_nac = inputs[Aircraft.Nacelle.FINENESS]

        d_eng = d_ref * np.sqrt(scale_fac)
        outputs[Aircraft.Nacelle.AVG_DIAMETER] = d_eng * d_nac_eng
        outputs[Aircraft.Nacelle.AVG_LENGTH] = (
            ld_nac * outputs[Aircraft.Nacelle.AVG_DIAMETER]
        )
        outputs[Aircraft.Nacelle.SURFACE_AREA] = (
            np.pi
            * outputs[Aircraft.Nacelle.AVG_DIAMETER]
            * outputs[Aircraft.Nacelle.AVG_LENGTH]
        )

    def compute_partials(self, inputs, J):
        d_ref = inputs[Aircraft.Engine.REFERENCE_DIAMETER]
        scale_fac = inputs[Aircraft.Engine.SCALE_FACTOR]
        d_nac_eng = inputs[Aircraft.Nacelle.CORE_DIAMETER_RATIO]
        ld_nac = inputs[Aircraft.Nacelle.FINENESS]

        tr = np.sqrt(scale_fac)
        d_eng = d_ref * tr
        d_nac = d_eng * d_nac_eng
        l_nac = d_nac * ld_nac

        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Engine.REFERENCE_DIAMETER] = tr * d_nac_eng
        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Engine.SCALE_FACTOR] = (
            d_nac_eng * d_ref / (2 * tr)
        )
        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Nacelle.CORE_DIAMETER_RATIO] = d_eng

        for wrt in [
            Aircraft.Engine.REFERENCE_DIAMETER,
            Aircraft.Engine.SCALE_FACTOR,
            Aircraft.Nacelle.CORE_DIAMETER_RATIO,
        ]:
            J[Aircraft.Nacelle.AVG_LENGTH, wrt] = (
                J[Aircraft.Nacelle.AVG_DIAMETER, wrt] * ld_nac
            )
            J[Aircraft.Nacelle.SURFACE_AREA, wrt] = np.pi * (
                J[Aircraft.Nacelle.AVG_DIAMETER, wrt] * l_nac
                + J[Aircraft.Nacelle.AVG_LENGTH, wrt] * d_nac
            )

        J[Aircraft.Nacelle.AVG_LENGTH, Aircraft.Nacelle.FINENESS] = d_nac
        J[Aircraft.Nacelle.SURFACE_AREA, Aircraft.Nacelle.FINENESS] = (
            np.pi * J[Aircraft.Nacelle.AVG_LENGTH, Aircraft.Nacelle.FINENESS] * d_nac
        )
