import numpy as np
import openmdao.api as om

from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Cold import atm_data as cold_210A
from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Hot import atm_data as hot_210A
from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Polar import atm_data as polar_210A
from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Tropical import atm_data as tropical_210A
from aviary.subsystems.atmosphere.data.StandardAtm1976 import atm_data as USatm1976
from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import AtmosphereModel, SpeedType
from aviary.variable_info.functions import add_aviary_option
from aviary.variable_info.variables import Dynamic, Settings


class Atmosphere(om.Group):
    """
    Group that contains atmospheric conditions for the aircraft's current flight
    condition, as well as conversions for different speed types (TAS, EAS, Mach).
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )

        self.options.declare(
            'h_def',
            values=('geopotential', 'geodetic'),
            default='geodetic',
            desc='The definition of altitude provided as input to the component. If '
            '"geodetic", it will be converted to geopotential based on Equation 19 in '
            'the original standard.',
        )

        self.options.declare(
            'input_speed_type',
            default=SpeedType.TAS,
            types=SpeedType,
            desc='defines input airspeed as equivalent airspeed, true airspeed, or mach number',
        )

        self.options.declare(
            'delta_T_Celcius',
            default=0.0,
            desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Celsius)',
        )

    def setup(self):
        nn = self.options['num_nodes']
        speed_type = self.options['input_speed_type']
        h_def = self.options['h_def']

        self.add_subsystem(
            name='standard_atmosphere',
            subsys=AtmosphereComp(num_nodes=nn, h_def=h_def),
            promotes_inputs=[Dynamic.Mission.ALTITUDE],
            promotes_outputs=['*'],
        )

        self.add_subsystem(
            name='flight_conditions',
            subsys=FlightConditions(num_nodes=nn, input_speed_type=speed_type),
            promotes=['*'],
        )


class AtmosphereComp(om.ExplicitComponent):
    """
    Component model for atmosphere tables.
    This model will calculate speed of sound and dynamic viscosity given inputs of
    akima splines for altitude, temperature, pressure, and density.

    Parameters
    ----------
    **kwargs : dict
        Dictionary of optional arguments.
    """

    def initialize(self):
        """Declare component options."""
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )
        self.options.declare(
            'h_def',
            values=('geopotential', 'geodetic'),
            default='geopotential',
            desc='The definition of altitude provided as input to the component.  If "geodetic",'
            'it will be converted to geopotential based on Equation 19 in the original standard.',
        )
        add_aviary_option(self, Settings.ATMOSPHERE_MODEL)
        self.options.declare(
            'delta_T_Celcius',
            types=(float, int),
            default=0.0,
            desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Celcius)',
        )

    def setup(self):
        """Add component inputs and outputs."""
        nn = self.options['num_nodes']

        self._dt = self.options['delta_T_Celcius']

        self._geodetic = self.options['h_def'] == 'geodetic'
        self._R0 = 6_356_766  # (meters) The effective Earth Radius
        # From the U.S. Standard Atmosphere 1976 publication located here
        # https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf

        gamma = 1.4  # Ratio of specific heads
        Rs = 8314.32  # J/(kmol*K), Gas constant
        M_air = 28.97  # (kg/kmol), molar mass of dry air
        self._R_air = Rs / M_air  # (J/ (kg * K)), gas constant for air
        self._K = gamma * Rs / M_air  # (J/(kg * K))

        self._S = 110.4  # (K) southerlands constant
        self._beta = 1.458e-6  # (s*m*K**(1/2))

        if self.options[Settings.ATMOSPHERE_MODEL] is AtmosphereModel.STANDARD:
            self.source_data = USatm1976
        elif self.options[Settings.ATMOSPHERE_MODEL] is AtmosphereModel.TROPICAL:
            self.source_data = tropical_210A
        elif self.options[Settings.ATMOSPHERE_MODEL] is AtmosphereModel.POLAR:
            self.source_data = polar_210A
        elif self.options[Settings.ATMOSPHERE_MODEL] is AtmosphereModel.HOT:
            self.source_data = hot_210A
        elif self.options[Settings.ATMOSPHERE_MODEL] is AtmosphereModel.COLD:
            self.source_data = cold_210A

        self.add_input(Dynamic.Mission.ALTITUDE, val=np.ones(nn), units='m')

        self.add_output(
            Dynamic.Atmosphere.TEMPERATURE, val=np.ones(nn), units='degK', desc='temperature of air'
        )
        self.add_output(
            Dynamic.Atmosphere.STATIC_PRESSURE, val=np.ones(nn), units='Pa', desc='pressure of air'
        )
        self.add_output(
            Dynamic.Atmosphere.DENSITY, val=np.ones(nn), units='kg/m**3', desc='density of air'
        )
        self.add_output(
            Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
            val=np.ones(nn),
            units='Pa*s',
            desc='dynamic viscosity of air',
        )
        self.add_output(
            Dynamic.Atmosphere.SPEED_OF_SOUND, val=np.ones(nn), units='m/s', desc='speed of sound'
        )
        self.add_output(
            'dsos_dh',
            val=np.ones(nn),
            units='1/s',
            desc='the change in the speed of sound with respect to height',
        )

        arange = np.arange(nn, dtype=int)
        self.declare_partials(
            [
                Dynamic.Atmosphere.TEMPERATURE,
                Dynamic.Atmosphere.STATIC_PRESSURE,
                Dynamic.Atmosphere.DENSITY,
                Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
                Dynamic.Atmosphere.SPEED_OF_SOUND,
                'dsos_dh',
            ],
            Dynamic.Mission.ALTITUDE,
            rows=arange,
            cols=arange,
        )

    def compute(self, inputs, outputs):
        """
        Interpolate atmospheric properties for a given altitude.

        Parameters
        ----------
        inputs : `Vector`
            `Vector` containing inputs.
        outputs : `Vector`
            `Vector` containing outputs.
        """
        table_points = self.source_data.alt
        h = inputs[Dynamic.Mission.ALTITUDE]

        if self._geodetic:
            h = (
                h / (self._R0 + h) * self._R0
            )  # Equation 19 from the U.S. Standard Atmosphere 1976 publication

        # From this point forward, h is geopotential altitude (z in the original reference).

        idx = np.searchsorted(table_points, h, side='left')
        h_bin_left = np.hstack((table_points[0], table_points))
        dx = h - h_bin_left[idx]

        coeffs = self.source_data.akima_T[idx]
        outputs[Dynamic.Atmosphere.TEMPERATURE] = T = (
            coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])) + self._dt
        )

        coeffs = self.source_data.akima_P[idx]
        outputs[Dynamic.Atmosphere.STATIC_PRESSURE] = pressure = coeffs[:, 0] + dx * (
            coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])
        )

        coeffs = self.source_data.akima_rho[idx]
        raw_density = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))

        # Equation 42, rho = (P * M)/(R * (T + dT))
        # Assumes pressure does not change (which is a simplification)
        # We know (P * M)/(R * T) from the akima table lookups (raw data)
        # We must correct the density from the lookup table by dt = delta_T_Celcius
        # Note : _R_air is R/M
        outputs[Dynamic.Atmosphere.DENSITY] = corrected_density = (
            raw_density ** (-1) + self._R_air * self._dt * pressure ** (-1)
        ) ** (-1)

        # Equation 50
        outputs[Dynamic.Atmosphere.SPEED_OF_SOUND] = (self._K * T) ** (0.5)

        # dsos_dh is only used for unsteady_solved_flight_conditions
        coeffs = self.source_data.akima_dT[idx]
        dT_dh = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        outputs['dsos_dh'] = 0.5 * (self._K * T) ** (-0.5) * dT_dh * self._K

        # Equation 51
        outputs[Dynamic.Atmosphere.DYNAMIC_VISCOSITY] = (
            self._beta * T ** (1.5) * (T + self._S) ** (-1)
        )

    def compute_partials(self, inputs, partials):
        """
        Compute sub-jacobian parts. The model is assumed to be in an unscaled state.

        Parameters
        ----------
        inputs : Vector
            Unscaled, dimensional input variables read via inputs[key].
        partials : Jacobian
            Subjac components written to partials[output_name, input_name].
        """
        table_points = self.source_data.alt
        h = inputs[Dynamic.Mission.ALTITUDE]
        dz_dh = 1.0

        if self._geodetic:
            dz_dh = (self._R0 / (self._R0 + h)) ** 2
            h = h / (self._R0 + h) * self._R0  # Equation 19 from the original standard.

        # From this point forward, h is geopotential altitude (z in the original reference).

        idx = np.searchsorted(table_points, h, side='left')
        h_index = np.hstack((table_points[0], table_points))
        dx = h - h_index[idx]

        coeffs = self.source_data.akima_T[idx]
        dT_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)
        T = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])) + self._dt

        coeffs = self.source_data.akima_P[idx]
        pressure = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        dP_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)

        coeffs = self.source_data.akima_rho[idx]
        raw_density = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        raw_drho_dh = coeffs[:, 1] + dx * (
            2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx
        )  # needs correction
        # corrected_density = (raw_density**(-1) + self._R_air*self._dt * pressure**(-1) )**(-1) # This gets complex because pressure changes as a function of h!
        corrected_drho_dh = (
            -1
            * (raw_density ** (-1) + self._R_air * self._dt * pressure ** (-1)) ** (-2)
            * (
                -1 * raw_density ** (-2) * raw_drho_dh
                + (-1 * self._R_air * self._dt * pressure ** (-2) * dP_dh)
            )
        )

        # outputs[Dynamic.Atmosphere.DYNAMIC_VISCOSITY] = self._beta * T**(1.5) * (T + self._S)**(-1)
        # need the product rule here
        dviscosity_dh = (
            1.5 * self._beta * T ** (0.5) * dT_dh * (T + self._S) ** (-1)
            + self._beta * T ** (1.5) * -1 * (T + self._S) ** (-2) * dT_dh
        )

        # sos = (self._K * T)**(0.5)
        # chain rule
        dsos_dh = 0.5 * (self._K * T) ** (-0.5) * self._K * dT_dh

        # similar to method in dymos
        coeffs = self.source_data.akima_dT[idx]
        d2T_dh2 = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)
        # dsos_dh = 0.5 * (self._K * T)**(-0.5) * dT_dh * self._K
        # product rule & chain rule
        partials['dsos_dh', Dynamic.Mission.ALTITUDE] = (
            -(0.5 * 0.5 * (self._K * T) ** (-1.5) * (self._K * dT_dh)) * (dT_dh * self._K)
            + 0.5 * (self._K * T) ** (-0.5) * d2T_dh2 * self._K
        )

        partials[Dynamic.Atmosphere.TEMPERATURE, Dynamic.Mission.ALTITUDE][...] = dT_dh.ravel()
        partials[Dynamic.Atmosphere.STATIC_PRESSURE, Dynamic.Mission.ALTITUDE][...] = dP_dh.ravel()
        partials[Dynamic.Atmosphere.DENSITY, Dynamic.Mission.ALTITUDE][...] = (
            corrected_drho_dh.ravel()
        )
        partials[Dynamic.Atmosphere.DYNAMIC_VISCOSITY, Dynamic.Mission.ALTITUDE][...] = (
            dviscosity_dh.ravel()
        )
        partials[Dynamic.Atmosphere.SPEED_OF_SOUND, Dynamic.Mission.ALTITUDE][...] = dsos_dh.ravel()

        if self._geodetic:
            partials[Dynamic.Atmosphere.TEMPERATURE, Dynamic.Mission.ALTITUDE][...] *= dz_dh
            partials[Dynamic.Atmosphere.STATIC_PRESSURE, Dynamic.Mission.ALTITUDE][...] *= dz_dh
            partials[Dynamic.Atmosphere.DENSITY, Dynamic.Mission.ALTITUDE][...] *= dz_dh
            partials[Dynamic.Atmosphere.DYNAMIC_VISCOSITY, Dynamic.Mission.ALTITUDE][...] *= dz_dh
            partials[Dynamic.Atmosphere.SPEED_OF_SOUND, Dynamic.Mission.ALTITUDE][...] *= dz_dh
            partials['dsos_dh', Dynamic.Mission.ALTITUDE] *= dz_dh**2
