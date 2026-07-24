import numpy as np
import openmdao.api as om

from aviary.variable_info.enums import Verbosity
from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Settings

epsilon = 0.05


def f(x):
    """Valid for x in [0.0, 1.0]."""
    diff = 0.5 - x
    y = 1.0 - np.arccos(2.0 * diff) / np.pi
    return y


def df(x):
    """First derivative of f(x), valid for x in (0.0, 1.0)."""
    diff = 0.5 - x
    dy = -2.0 / np.sqrt(1.0 - 4 * diff * diff) / np.pi
    return dy


def d2f(x):
    """Second derivative of f(x), valid for x in (0.0, 1.0)."""
    diff = 0.5 - x
    d2y = 8.0 * diff / np.sqrt(1.0 - 4 * diff * diff) / (1.0 - 4 * diff * diff) / np.pi
    return d2y


def g1(x):
    """
    Returns a cubic function g1(x) such that:
    g1(0) = 1
    g1(ε) = f(ε)
    g1'(ε) = f'(ε)
    g1"(ε) = f"(ε).
    """
    A1 = f(epsilon)
    B1 = df(epsilon)
    C1 = d2f(epsilon)
    d1 = (A1 - 1 - epsilon * B1 + 0.5 * epsilon**2 * C1) / epsilon**3
    c1 = (C1 - 6 * d1 * epsilon) / 2
    b1 = B1 - epsilon * C1 + 3 * d1 * epsilon**2
    a1 = 1
    y = a1 + b1 * x + c1 * x**2 + d1 * x**3
    return y


def dg1(x):
    """First derivative of g1(x)."""
    A1 = f(epsilon)
    B1 = df(epsilon)
    C1 = d2f(epsilon)
    d1 = (A1 - 1 - epsilon * B1 + 0.5 * epsilon**2 * C1) / epsilon**3
    c1 = (C1 - 6 * d1 * epsilon) / 2
    b1 = B1 - epsilon * C1 + 3 * d1 * epsilon**2
    dy = b1 + 2 * c1 * x + 3 * d1 * x**2
    return dy


def g2(x):
    """
    Returns a cubic function g2(x) such that:
    g2(1) = 0
    g2(ε) = f(1-ε)
    g2'(ε) = f'(1-ε)
    g2"(ε) = f"(1-ε).
    """
    delta = 1.0 - epsilon
    A2 = f(delta)
    B2 = df(delta)
    C2 = d2f(delta)
    d2 = -(A2 + B2 * epsilon + 0.5 * C2 * epsilon**2) / epsilon**3
    c2 = (C2 - 6.0 * d2 * delta) / 2.0
    b2 = B2 - C2 * delta + 3.0 * d2 * delta**2
    a2 = -(b2 + c2 + d2)
    y = a2 + b2 * x + c2 * x**2 + d2 * x**3
    return y


def dg2(x):
    """First derivative of g2(x)."""
    delta = 1.0 - epsilon
    A2 = f(delta)
    B2 = df(delta)
    C2 = d2f(delta)
    d2 = -(A2 + B2 * epsilon + 0.5 * C2 * epsilon**2) / epsilon**3
    c2 = (C2 - 6.0 * d2 * delta) / 2.0
    b2 = B2 - C2 * delta + 3.0 * d2 * delta**2
    dy = b2 + 2 * c2 * x + 3.0 * d2 * x**2
    return dy


class PercentNotInFuselage(om.ExplicitComponent):
    """
    For BWB, engine may be (partially) buried into fuselage. Compute the percentage of
    corresponding surface area of nacelles not buried in fuselage.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(
            self,
            Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE,
            shape=num_engine_type,
            units='unitless',
        )

        self.add_output('percent_exposed', shape=num_engine_type, units='unitless')

    def setup_partials(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            'percent_exposed',
            [Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE],
            rows=shape,
            cols=shape,
        )

    def compute(self, inputs, outputs):
        x = inputs[Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE]
        if x >= epsilon and x <= 1 - epsilon:
            diff = 0.5 - x
            pct_swn = 1.0 - np.arccos(2.0 * diff) / np.pi
        elif x >= 0.0 and x < epsilon:
            pct_swn = g1(x)
        elif x <= 1.0 and x > 1 - epsilon:
            pct_swn = g2(x)
        else:
            raise om.AnalysisError(
                'The parameter Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE is out of range.'
            )

        outputs['percent_exposed'] = pct_swn

    def compute_partials(self, inputs, J):
        x = inputs[Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE]
        if x >= epsilon and x <= 1 - epsilon:
            diff = 0.5 - x
            d_pct_swn = -2.0 / np.sqrt(1.0 - 4 * diff * diff) / np.pi
        elif x >= 0.0 and x < epsilon:
            d_pct_swn = dg1(x)
        elif x <= 1.0 and x > 1 - epsilon:
            d_pct_swn = dg2(x)
        else:
            raise om.AnalysisError(
                'The parameter Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE is out of range.'
            )

        J['percent_exposed', Aircraft.Nacelle.PERCENT_DIAM_BURIED_IN_FUSELAGE] = d_pct_swn


class GASPEngineDiameter(om.ExplicitComponent):
    """
    Engine geometry calculation based on engine diameter. It returns Aircraft.Nacelle.AVG_DIAMETER,
    It follows the algorithm in GASP.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(
            self, Aircraft.Engine.INLET_AREA_COEFFICIENT, shape=num_engine_type, units='unitless'
        )
        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(
            self, Aircraft.Nacelle.CORE_DIAMETER_RATIO, shape=num_engine_type, units='unitless'
        )

        add_aviary_output(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')

    def setup_partials(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)
        nn = np.zeros(num_engine_type, dtype=int)

        self.declare_partials(
            Aircraft.Nacelle.AVG_DIAMETER,
            [
                Aircraft.Design.GROSS_MASS,
            ],
            rows=shape,
            cols=nn,
        )
        self.declare_partials(
            Aircraft.Nacelle.AVG_DIAMETER,
            [
                Aircraft.Engine.INLET_AREA_COEFFICIENT,
                Aircraft.Nacelle.CORE_DIAMETER_RATIO,
            ],
            rows=shape,
            cols=shape,
        )

    def compute(self, inputs, outputs):
        num_engine = self.options[Aircraft.Engine.NUM_ENGINES]
        coeff_inlet = inputs[Aircraft.Engine.INLET_AREA_COEFFICIENT]

        gross_mass = inputs[Aircraft.Design.GROSS_MASS]
        core_diam_ratio = inputs[Aircraft.Nacelle.CORE_DIAMETER_RATIO]

        area_engine = coeff_inlet * gross_mass / num_engine
        diam_engine = np.sqrt(4.0 * area_engine / np.pi)
        diam_nacelle = core_diam_ratio * diam_engine

        outputs[Aircraft.Nacelle.AVG_DIAMETER] = diam_nacelle

    def compute_partials(self, inputs, J):
        num_engine = self.options[Aircraft.Engine.NUM_ENGINES]
        coeff_inlet = inputs[Aircraft.Engine.INLET_AREA_COEFFICIENT]

        gross_mass = inputs[Aircraft.Design.GROSS_MASS]
        gross_mass = np.full(len(num_engine), gross_mass)
        core_diam_ratio = inputs[Aircraft.Nacelle.CORE_DIAMETER_RATIO]

        area_engine = coeff_inlet * gross_mass / num_engine
        diam_engine = np.sqrt(4.0 * area_engine / np.pi)

        darea_engine_dgross_mass = coeff_inlet * np.ones(1) / num_engine
        ddiam_engine_dgross_mass = 2 / np.pi / diam_engine * darea_engine_dgross_mass
        ddiam_nacelle_dgross_mass = core_diam_ratio * ddiam_engine_dgross_mass
        ddiam_nacelle_dcore_diam_ratio = diam_engine

        ddiam_nacelle_dcoeff_inlet = (
            core_diam_ratio * 2 / diam_engine * gross_mass / num_engine / np.pi
        )

        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Design.GROSS_MASS] = ddiam_nacelle_dgross_mass
        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Nacelle.CORE_DIAMETER_RATIO] = (
            ddiam_nacelle_dcore_diam_ratio
        )
        J[Aircraft.Nacelle.AVG_DIAMETER, Aircraft.Engine.INLET_AREA_COEFFICIENT] = (
            ddiam_nacelle_dcoeff_inlet
        )


class GASPEngineLength(om.ExplicitComponent):
    """
    Engine geometry calculation based on engine diameter. It returns Nacelle.AVG_LENGTH.
    It follows the algorithm in GASP.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        add_aviary_input(self, Aircraft.Design.GROSS_MASS, units='lbm')
        add_aviary_input(
            self, Aircraft.Nacelle.CORE_DIAMETER_RATIO, shape=num_engine_type, units='unitless'
        )
        add_aviary_input(self, Aircraft.Nacelle.FINENESS, shape=num_engine_type, units='unitless')
        self.add_input('percent_exposed', val=np.ones(num_engine_type), units='unitless')
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')

        add_aviary_output(self, Aircraft.Nacelle.AVG_LENGTH, shape=num_engine_type, units='ft')

    def setup_partials(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Nacelle.AVG_LENGTH,
            [
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Nacelle.FINENESS,
            ],
            rows=shape,
            cols=shape,
        )

    def compute(self, inputs, outputs):
        fineness_nac = inputs[Aircraft.Nacelle.FINENESS]
        diam_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        len_nacelle = fineness_nac * diam_nacelle

        outputs[Aircraft.Nacelle.AVG_LENGTH] = len_nacelle

    def compute_partials(self, inputs, J):
        fineness_nac = inputs[Aircraft.Nacelle.FINENESS]
        diam_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]

        J[Aircraft.Nacelle.AVG_LENGTH, Aircraft.Nacelle.AVG_DIAMETER] = fineness_nac
        J[Aircraft.Nacelle.AVG_LENGTH, Aircraft.Nacelle.FINENESS] = diam_nacelle


class GASPEngineSurfaceArea(om.ExplicitComponent):
    """
    Engine geometry calculation based on engine diameter. It returns Aircraft.Nacelle.SURFACE_AREA.
    It follows the algorithm in GASP.
    """

    def initialize(self):
        add_aviary_option(self, Aircraft.Engine.NUM_ENGINES)
        add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])

        self.add_input('percent_exposed', val=np.ones(num_engine_type), units='unitless')
        add_aviary_input(self, Aircraft.Nacelle.AVG_DIAMETER, shape=num_engine_type, units='ft')
        add_aviary_input(self, Aircraft.Nacelle.AVG_LENGTH, shape=num_engine_type, units='ft')

        add_aviary_output(self, Aircraft.Nacelle.SURFACE_AREA, shape=num_engine_type, units='ft**2')

    def setup_partials(self):
        num_engine_type = len(self.options[Aircraft.Engine.NUM_ENGINES])
        shape = np.arange(num_engine_type)

        self.declare_partials(
            Aircraft.Nacelle.SURFACE_AREA,
            [
                Aircraft.Nacelle.AVG_DIAMETER,
                Aircraft.Nacelle.AVG_LENGTH,
                'percent_exposed',
            ],
            rows=shape,
            cols=shape,
        )

    def compute(self, inputs, outputs):
        pct_exposed = inputs['percent_exposed']
        diam_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        len_nacelle = inputs[Aircraft.Nacelle.AVG_LENGTH]
        wet_area_nacelle = np.pi * diam_nacelle * len_nacelle * pct_exposed

        outputs[Aircraft.Nacelle.SURFACE_AREA] = wet_area_nacelle

    def compute_partials(self, inputs, J):
        pct_exposed = inputs['percent_exposed']
        diam_nacelle = inputs[Aircraft.Nacelle.AVG_DIAMETER]
        len_nacelle = inputs[Aircraft.Nacelle.AVG_LENGTH]

        J[Aircraft.Nacelle.SURFACE_AREA, Aircraft.Nacelle.AVG_DIAMETER] = (
            np.pi * len_nacelle * pct_exposed
        )
        J[Aircraft.Nacelle.SURFACE_AREA, Aircraft.Nacelle.AVG_LENGTH] = (
            np.pi * diam_nacelle * pct_exposed
        )
        J[Aircraft.Nacelle.SURFACE_AREA, 'percent_exposed'] = np.pi * diam_nacelle * len_nacelle


class GASPEngineSizeGroup(om.Group):
    def setup(self):
        self.add_subsystem(
            'perc',
            PercentNotInFuselage(),
            promotes_inputs=['*'],
            promotes_outputs=['percent_exposed'],
        )

        self.add_subsystem(
            'eng_diameter',
            GASPEngineDiameter(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'eng_length',
            GASPEngineLength(),
            promotes_inputs=['*'],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            'eng_area',
            GASPEngineSurfaceArea(),
            promotes_inputs=['*'] + ['percent_exposed'],
            promotes_outputs=['*'],
        )
