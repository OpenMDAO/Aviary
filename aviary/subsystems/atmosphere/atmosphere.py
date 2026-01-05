import openmdao.api as om

from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic

import numpy as np

from aviary.subsystems.atmosphere.StandardAtm1976 import atm_data as USatm1976
from aviary.subsystems.atmosphere.MIL_SPEC_210A_Tropical import atm_data as tropical_210A
from aviary.subsystems.atmosphere.MIL_SPEC_210A_Polar import atm_data as polar_210A
from aviary.subsystems.atmosphere.MIL_SPEC_210A_Hot import atm_data as hot_210A
from aviary.subsystems.atmosphere.MIL_SPEC_210A_Cold import atm_data as cold_210A


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
            'delta_T_Kelvin',
            default=0.0,
            desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Kelvine)',
        )

        self.options.declare(
            'data_source',
            default='USatm1976',
            desc='The atmospheric model used. Chose one of USatm1976, tropical, polar, hot, cold.',
        )

    def setup(self):
        nn = self.options['num_nodes']
        speed_type = self.options['input_speed_type']
        h_def = self.options['h_def']

        self.add_subsystem(
            name='standard_atmosphere',
            subsys=AtmosphereComp(num_nodes=nn, h_def=h_def),
            promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                '*',
                ('sos', Dynamic.Atmosphere.SPEED_OF_SOUND),
                ('rho', Dynamic.Atmosphere.DENSITY),
                ('temp', Dynamic.Atmosphere.TEMPERATURE),
                ('pres', Dynamic.Atmosphere.STATIC_PRESSURE),
            ],
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
        """
        Declare component options.
        """
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
        self.options.declare(
            'data_source',
            values=('USatm1976', 'tropical', 'polar', 'hot', 'cold'),
            default='USatm1976',
            desc='The atmospheric model to use as source data.',
        )
        self.options.declare(
            'delta_T_Kelvin',
            types=(float, int),
            default=0.0,
            desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Kelvin)',
        )

    def setup(self):
        """
        Add component inputs and outputs.
        """
        nn = self.options['num_nodes']

        self._dt = self.options['delta_T_Kelvin']

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

        if self.options['data_source'] == 'USatm1976':
            self.source_data = USatm1976
        elif self.options['data_source'] == 'tropical':
            self.source_data = tropical_210A
        elif self.options['data_source'] == 'polar':
            self.source_data = polar_210A
        elif self.options['data_source'] == 'hot':
            self.source_data = hot_210A
        elif self.options['data_source'] == 'cold':
            self.source_data = cold_210A
        else:
            Warning(
                'User has specified unknown atmosphere model. Please use one of: USatm1976, tropical, polar, hot, cold'
            )

        self.add_input('h', val=1.0 * np.ones(nn), units='m')

        self.add_output('temp', val=1.0 * np.ones(nn), units='degK', desc='temperature of air')
        self.add_output('pres', val=1.0 * np.ones(nn), units='Pa', desc='pressure of air')
        self.add_output('rho', val=1.0 * np.ones(nn), units='kg/m**3', desc='density of air')
        self.add_output(
            'viscosity', val=1.0 * np.ones(nn), units='Pa*s', desc='dynamic viscosity of air'
        )
        self.add_output('sos', val=1 * np.ones(nn), units='m/s', desc='speed of sound')
        self.add_output('dsos_dh', val=1 * np.ones(nn), units='1/s', desc='the change in the speed of sound with respect to height')

        arange = np.arange(nn, dtype=int)
        self.declare_partials(
            ['temp', 'pres', 'rho', 'viscosity', 'sos', 'dsos_dh'], 'h', rows=arange, cols=arange
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
        h = inputs['h']

        if self._geodetic:
            h = (
                h / (self._R0 + h) * self._R0
            )  # Equation 19 from the U.S. Standard Atmosphere 1976 publication

        # From this point forward, h is geopotential altitude (z in the original reference).

        idx = np.searchsorted(table_points, h, side='left')
        h_bin_left = np.hstack((table_points[0], table_points))
        dx = h - h_bin_left[idx]

        coeffs = self.source_data.akima_T[idx]
        outputs['temp'] = temp = (
            coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])) + self._dt
        )

        coeffs = self.source_data.akima_P[idx]
        outputs['pres'] = pressure = coeffs[:, 0] + dx * (
            coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])
        )

        coeffs = self.source_data.akima_rho[idx]
        raw_density = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))

        # Equation 42, rho = (P * M)/(R * (T + dT))
        # Assumes pressure does not change (which is a simplification)
        # We know (P * M)/(R * T) from the akima table lookups (raw data)
        # We must correct the density from the lookup table by dt = delta_T_Kelvin
        outputs['rho'] = corrected_density = (
            raw_density ** (-1) + self._R_air * self._dt * pressure ** (-1)
        ) ** (-1)

        # Equation 50
        outputs['sos'] = (self._K * temp) ** (0.5)

        # dsos_dh is only used for unsteady_solved_flight_conditions
        coeffs = self.source_data.akima_dT[idx]
        dT_dh = (coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])))
        outputs['dsos_dh'] = (0.5 * (self._K * temp) ** (-0.5) * dT_dh * self._K)

        # Equation 51
        outputs['viscosity'] = self._beta * temp ** (1.5) * (temp + self._S) ** (-1)

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
        h = inputs['h']
        dz_dh = 1.0

        if self._geodetic:
            dz_dh = (self._R0 / (self._R0 + h)) ** 2
            h = h / (self._R0 + h) * self._R0  # Equation 19 from the original standard.

        # From this point forward, h is geopotential altitude (z in the original reference).

        idx = np.searchsorted(table_points, h, side='left')
        h_index = np.hstack((table_points[0], table_points))
        dx = h - h_index[idx]

        coeffs = self.source_data.akima_T[idx]
        temp = (
            coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])) + self._dt
        )
        dT_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)

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

        # outputs['viscosity'] = self._beta * temp**(1.5) * (temp + self._S)**(-1)
        # need the product rule here
        dviscosity_dh = (
            1.5 * self._beta * temp ** (0.5) * dT_dh * (temp + self._S) ** (-1)
            + self._beta * temp ** (1.5) * -1 * (temp + self._S) ** (-2) * dT_dh
        )

        # sos = (self._K * temp)**(0.5)
        # chain rule
        dsos_dh = 0.5 * (self._K * temp) ** (-0.5) * self._K * dT_dh

        # similar to method in dymos
        coeffs2 = self.source_data.akima_dT[idx]
        d2T_dh2 = (coeffs2[:, 1] + dx * (2.0 * coeffs2[:, 2] + 3.0 * coeffs2[:, 3] * dx))
        partials['dsos_dh', 'h'] = 0.5 * np.sqrt(self._K / temp) * (d2T_dh2 - 0.5 * dT_dh**2 / temp)

        partials['temp', 'h'][...] = dT_dh.ravel()
        partials['pres', 'h'][...] = dP_dh.ravel()
        partials['rho', 'h'][...] = corrected_drho_dh.ravel()
        partials['viscosity', 'h'][...] = dviscosity_dh.ravel()
        partials['sos', 'h'][...] = dsos_dh.ravel()

        if self._geodetic:
            partials['temp', 'h'][...] *= dz_dh
            partials['pres', 'h'][...] *= dz_dh
            partials['rho', 'h'][...] *= dz_dh  # does this still apply?
            partials['viscosity', 'h'][...] *= dz_dh  # does this still apply?
            partials['sos', 'h'][...] *= dz_dh  # does this still apply?
            partials['dsos_dh', 'h'] *= dz_dh ** 2 # does this still apply?


def _build_akima_coefs(out_stream, raw_data, units):
    """
    Print out the Akima coefficients based on the raw atmospheric data.

    This is used to more rapidly interpolate the data and the rate of change of rho wrt altitude.

    Inputs
    -------
    units: Float ('SI', or 'English')
        Describes the input units in either SI or English.
        If SI units are selected then the data should be input as:
            (altitude: m, temp: degK, pressure: mb, density: kg/m**3)
        If English units are selected then the data should be input as:
            (altitude: ft, temp: degF, pressure: inHg60, density: lbm/ft**3)

    Returns
    -------
    dict
        A mapping of the variable name and Akima coeffcient values for each table in the atmosphere.
        Output units are always in SI.
        (altitude: m, temp: degK, pressure: Pa, density: kg/ft**3)
    """

    raw_data = np.reshape(raw_data, (raw_data.size // 4, 4))

    from collections import namedtuple

    atm_data = namedtuple('atm_data', ['alt', 'temp', 'pres', 'rho'])

    atm_data.alt = raw_data[:, 0]

    atm_data.T = raw_data[:, 1]

    atm_data.P = raw_data[:, 2]

    atm_data.rho = raw_data[:, 3]

    # Covert all data to SI units
    if units == 'SI':
        atm_data.P *= 100  # mb -> pascal
    elif units == 'English':
        atm_data.alt *= 0.3048  # ft -> m
        atm_data.T = (atm_data.T - 32) * 5 / 9 + 273.15  # degF -> degK
        atm_data.P *= 3376.85  # inHg60 -> Pascal
        atm_data.rho *= 0.453592 / (0.3048**3)  # lbm/ft**3 -> kg/m**3
    else:
        print(f"units must be SI or English but '{units}' was supplied.")
        exit()
    # Units have now been translated into (altitude (meters), temperature (degK), pressure(pascals), dynamic viscosity (kg/m**3))

    import textwrap
    from openmdao.components.interp_util.interp import InterpND

    coeff_data = {}

    T_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.T, extrapolate=True)
    P_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.P, extrapolate=True)
    rho_interp = InterpND(
        method='1D-akima', points=atm_data.alt, values=atm_data.rho, extrapolate=True
    )

    _, _dT_dh = T_interp.interpolate(atm_data.alt, compute_derivative=True)
    dT_interp = InterpND(method='1D-akima', points=atm_data.alt, values=_dT_dh.ravel(), extrapolate=True)

    # Find midpoints of all bins plus an extrapolation point on each end.
    min_alt = np.min(atm_data.alt)
    max_alt = np.max(atm_data.alt)

    # We need to compute coeffs in the "extrapolation bins" as well, so append these.
    h = np.hstack((min_alt - 5000, atm_data.alt, max_alt + 5000))
    hbin = h[:-1] + 0.5 * np.diff(h)
    n = len(hbin)

    coeffs_T = np.empty((n, 4))
    coeffs_P = np.empty((n, 4))
    coeffs_rho = np.empty((n, 4))
    coeffs_dT = np.empty((n, 4))

    interps = [T_interp, P_interp, rho_interp, dT_interp]
    coeff_arrays = [coeffs_T, coeffs_P, coeffs_rho, coeffs_dT]

    np.set_printoptions(precision=18)

    with np.printoptions(linewidth=100, threshold=np.inf):
        # Print altitude in correct units:
        if out_stream is not None:
            print(f'atm_data.alt = \\', file=out_stream)
            print(
                textwrap.indent(repr(atm_data.alt).replace('array', 'np.array'), '    '),
                file=out_stream,
            )
            print('', file=out_stream)
        input('Press Enter to continue: ')

    vars = ['T', 'P', 'rho', 'dT']
    with np.printoptions(linewidth=1024, threshold=np.inf):
        # Print akima splines in correct units
        for var, interp, coeff_array in zip(vars, interps, coeff_arrays):
            _ = interp.interpolate(hbin, compute_derivative=False)
            coeff_cache = interp.table.vec_coeff

            for i in range(n):
                a, b, c, d = coeff_cache[i]
                coeff_array[i, 0] = a
                coeff_array[i, 1] = b
                coeff_array[i, 2] = c
                coeff_array[i, 3] = d

            if out_stream is not None:
                print(f'atm_data.akima_{var} = \\', file=out_stream)
                print(
                    textwrap.indent(repr(coeff_array).replace('array', 'np.array'), '    '),
                    file=out_stream,
                )
                print('', file=out_stream)

            coeff_data[f'atm_data.akima_{var}'] = coeff_array
            input('Press Enter to continue: ')
    print('Program Complete')

    return coeff_data


if __name__ == '__main__':
    build_akima = False
    test_values = True

    if build_akima:
        ############### Generate Akima Splines Below ################
        # Running this script generates and prints the Akima coefficients using the OpenMDAO akima1D interpolant.

        print(
            'WARNING: _build_akima_coefs() does not have the standard unit conversion capabilities you may be used to from OpenMDAO. '
            'Make sure your input units match the requirements shown in _build_akima_coefs()!'
        )
        input('Press Enter to continue: ')

        from aviary.subsystems.atmosphere.MIL_SPEC_210A_Polar import (
            _raw_data,
        )  # replace this with your new raw data

        import sys

        _build_akima_coefs(out_stream=sys.stdout, raw_data=_raw_data, units='English')

    if test_values:
        ################ Test problem below ################

        prob = om.Problem()

        # 'USatm1976', 'tropical', 'polar', 'hot', 'cold'
        atm_model = prob.model.add_subsystem(
            'comp',
            AtmosphereComp(data_source='tropical', delta_T_Kelvin=0, num_nodes=6),
            promotes=['*'],
        )

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        prob.set_val('h', [0, 10000, 35000, 55000, 70000, 100000], units='ft')

        prob.run_model()

        #prob.check_partials(method='cs')

        # print('Temperatures (K):', prob.get_val('temp', units='K'))
        # print('Pressure (Pa)', prob.get_val('pres', units='Pa'))
        # print('Density (kg/m**3)', prob.get_val('rho', units='kg/m**3'))
        # print('Viscosity (Pa*s)', prob.get_val('viscosity', units='Pa*s'))
        # print('Speed of Sound (m/s)', prob.get_val('sos', units='m/s'))

        print('Temperatures (degF):', prob.get_val('temp', units='degF'))
        print('Pressure (inHg60)', prob.get_val('pres', units='inHg60'))
        print('Density (lbm/ft**3)', prob.get_val('rho', units='lbm/ft**3'))
        print('Viscosity (Pa*s)', prob.get_val('viscosity', units='Pa*s'))
        print('Speed of Sound (m/s)', prob.get_val('sos', units='m/s'))
