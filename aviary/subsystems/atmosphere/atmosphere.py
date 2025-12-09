import openmdao.api as om

from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic

import numpy as np

from StandardAtm1976 import atm_data as USatm1976
from MIL_SPEC_210A_Tropical import atm_data as tropical_210A
from MIL_SPEC_210A_Polar import atm_data as polar_210A
from MIL_SPEC_210A_Hot import atm_data as hot_210A
from MIL_SPEC_210A_Cold import atm_data as cold_210A

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
            'isa_delta_T_Kelvin',
            default=0.0,
            desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Kelvine)',
        )

        self.options.declare(
            'data_source',
            default='USatm1976',
            desc='The atmospheric model used. Chose one of USatm1976, tropical, polar, hot, cold.'
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
        self.options.declare('num_nodes', types=int,
                             desc='Number of nodes to be evaluated in the RHS')

        gamma = 1.4  # Ratio of specific heads
        gas_c = 1716.49  # Gas constant (ft lbf)/(slug R)
        self._K = gamma * gas_c

        self.options.declare('h_def', values=('geopotential', 'geodetic'), default='geopotential',
                             desc='The definition of altitude provided as input to the component.  If "geodetic",'
                                  'it will be converted to geopotential based on Equation 19 in the original standard.')
        self.options.declare('data_source', values=('USatm1976', 'tropical', 'polar', 'hot', 'cold'), default='USatm1976',
                             desc='The atmospheric model to use as source data.')
        self.options.declare('isa_delta_T_Kelvin', types=float, default=0.,
                             desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Kelvin)')
        
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
            Warning('User has specified unknown atmosphere model. Please use one of: USatm1976, tropical, polar, hot, cold')

    def setup(self):
        """
        Add component inputs and outputs.
        """
        nn = self.options['num_nodes']

        self._dt = self.options['isa_delta_T_Kelvin']

        self._geodetic = self.options['h_def'] == 'geodetic'
        self._R0 = 6_356_766 # (meters) The effective Earth Radius
        # From the U.S. Standard Atmosphere 1976 publication located here
        # https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf

        gamma = 1.4  # Ratio of specific heads
        Rs = 8314.32 # J/(kmol*K), Gas constant 
        M_air = 28.97 # (kg/kmol), molar mass of dry air
        self._R_air = Rs/M_air # (J/ (kg * K)), gas constant for air
        self._K = gamma * Rs / M_air #(J/(kg * K))

        self._S = 110.4 #(K) southerlands constant
        self._beta = 1.458e-6 #(s*m*K^(1/2))

        self.add_input('h', val=1. * np.ones(nn), units='m')

        self.add_output('temp', val=1. * np.ones(nn), units='degK', desc='temperature of air')
        self.add_output('pres', val=1. * np.ones(nn), units='Pa', desc='pressure of air')
        self.add_output('rho', val=1. * np.ones(nn), units='kg/m^3', desc='density of air')
        self.add_output('viscosity', val=1. * np.ones(nn), units='Pa*sec', desc='dynamic viscosity of air')
        self.add_output('sos', val=1 * np.ones(nn), units='m/s', desc='speed of sound')

        arange = np.arange(nn, dtype=int)
        self.declare_partials(['temp', 'pres', 'rho', 'viscosity', 'sos'], 'h',
                              rows=arange, cols=arange)

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
            h = h / (self._R0 + h) * self._R0  # Equation 19 from the U.S. Standard Atmosphere 1976 publication

        # From this point forward, h is geopotential altitude (z in the original reference).

        idx = np.searchsorted(table_points, h, side='left')
        h_bin_left = np.hstack((table_points[0], table_points))
        dx = h - h_bin_left[idx]

        coeffs = self.source_data.akima_T[idx]
        outputs['temp'] = temp = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3])) + self._dt

        coeffs = self.source_data.akima_P[idx]
        outputs['pres'] = pressure = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))

        coeffs = self.source_data.akima_rho[idx]
        raw_density = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))

        # Equation 42, rho = (P * M)/(R * (T + dT))
        # Assumes pressure does not change (which is a simplification)
        # We know (P * M)/(R * T) from the akima table lookups (raw data)
        # We must correct the density from the lookup table by dt = isa_delta_T_Kelvin
        outputs['rho'] = corrected_density = 1 / (raw_density + self._R_air*self._dt / pressure )

        # Equation 50
        outputs['sos'] = np.sqrt(self._K * outputs['temp'])

        # Equation 51
        outputs['viscosity'] = self._beta * temp^(1.5) / (temp + self._S)

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
        dT_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)
        T = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))

        coeffs = self.source_data.akima_P[idx]
        dP_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)

        coeffs = self.source_data.akima_rho[idx]
        drho_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)

        coeffs = USatm1976Data.akima_viscosity[idx]
        dvisc_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)

        coeffs = USatm1976Data.akima_drho[idx]
        d2rho_dh2 = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)

        partials['temp', 'h'][...] = dT_dh.ravel()
        partials['pres', 'h'][...] = dP_dh.ravel()
        partials['rho', 'h'][...] = drho_dh.ravel()
        partials['viscosity', 'h'][...] = dvisc_dh.ravel()
        partials['drhos_dh', 'h'][...] = d2rho_dh2.ravel()
        partials['sos', 'h'][...] = (0.5 / np.sqrt(self._K * T) * partials['temp', 'h'] * self._K)

        if self._geodetic:
            partials['sos', 'h'][...] *= dz_dh
            partials['temp', 'h'][...] *= dz_dh
            partials['viscosity', 'h'][...] *= dz_dh
            partials['rho', 'h'][...] *= dz_dh
            partials['pres', 'h'][...] *= dz_dh
            partials['drhos_dh', 'h'][...] *= dz_dh ** 2


def _build_akima_coefs(out_stream=sys.stdout):
    """
    Print out the Akima coefficients based on the raw atmospheric data.

    This is used to more rapidly interpolate the data and the rate of change of rho wrt altitude.

    Returns
    -------
    dict
        A mapping of the variable name and Akima coeffcient values for each table in the atmosphere.
    """
    import textwrap
    from openmdao.components.interp_util.interp import InterpND

    coeff_data = {}

    T_interp = InterpND(method='1D-akima', points=USatm1976Data.alt, values=USatm1976Data.T, extrapolate=True)
    P_interp = InterpND(method='1D-akima', points=USatm1976Data.alt, values=USatm1976Data.P, extrapolate=True)
    rho_interp = InterpND(method='1D-akima', points=USatm1976Data.alt, values=USatm1976Data.rho, extrapolate=True)
    visc_interp = InterpND(method='1D-akima', points=USatm1976Data.alt, values=USatm1976Data.viscosity,
                           extrapolate=True)

    _, _drho_dh = rho_interp.interpolate(USatm1976Data.alt, compute_derivative=True)
    drho_interp = InterpND(method='1D-akima', points=USatm1976Data.alt, values=_drho_dh.ravel(), extrapolate=True)

    _, _dT_dh = T_interp.interpolate(USatm1976Data.alt, compute_derivative=True)
    dT_interp = InterpND(method='1D-akima', points=USatm1976Data.alt, values=_dT_dh.ravel(), extrapolate=True)

    # Find midpoints of all bins plus an extrapolation point on each end.
    min_alt = np.min(USatm1976Data.alt)
    max_alt = np.max(USatm1976Data.alt)

    # We need to compute coeffs in the "extrapolation bins" as well, so append these.
    h = np.hstack((min_alt - 5000, USatm1976Data.alt, max_alt + 5000))
    hbin = h[:-1] + 0.5 * np.diff(h)
    n = len(hbin)

    coeffs_T = np.empty((n, 4))
    coeffs_P = np.empty((n, 4))
    coeffs_rho = np.empty((n, 4))
    coeffs_visc = np.empty((n, 4))
    coeffs_drho = np.empty((n, 4))
    coeffs_dT = np.empty((n, 4))

    interps = [T_interp, P_interp, rho_interp, visc_interp, drho_interp, dT_interp]
    coeff_arrays = [coeffs_T, coeffs_P, coeffs_rho, coeffs_visc, coeffs_drho, coeffs_dT]

    np.set_printoptions(precision=18)
    vars = ['T', 'P', 'rho', 'viscosity', 'drho', 'dT']
    with np.printoptions(linewidth=1024):
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
                print(f'USatm1976Data.akima_{var} = \\', file=out_stream)
                print(textwrap.indent(repr(coeff_array).replace('array', 'np.array'), '    '),
                      file=out_stream)
                print('', file=out_stream)

            coeff_data[f'USatm1976Data.akima_{var}'] = coeff_array
            input("Press Enter to continue: ")
    print("Program Complete")

    return coeff_data


if __name__ == "__main__":
    # Running this script generates and prints the Akima coefficients using the OpenMDAO akima1D interpolant.

    from Aviary.aviary.subsystems.atmosphere.atmosphereComp import _raw_data # replace this with your new raw data

    import sys
    _build_akima_coefs(raw_data=_raw_data, out_stream=sys.stdout)