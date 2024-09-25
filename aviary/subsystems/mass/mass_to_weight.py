import numpy as np
import openmdao.api as om
from aviary.constants import GRAV_ENGLISH_LBM


class MassToWeight(om.ExplicitComponent):
    """
    Component to convert mass to weight.
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int, default=1)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input(
            "mass",
            val=np.ones(nn),
            units="lbm",
            desc="mass of the aircraft",
        )

        self.add_output(
            "weight",
            val=np.ones(nn),
            units="lbf",
            desc="weight of the aircraft",
        )

    def setup_partials(self):
        nn = self.options['num_nodes']
        arange = np.arange(nn)
        self.declare_partials("weight",
                              "mass",
                              rows=arange,
                              cols=arange,
                              val=np.full(nn, GRAV_ENGLISH_LBM)
                              )

    def compute(self, inputs, outputs):
        outputs["weight"] = inputs["mass"] * GRAV_ENGLISH_LBM
