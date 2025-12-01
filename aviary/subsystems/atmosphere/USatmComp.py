"""
United States standard atmosphere 1976 tables, data
obtained from http://www.digitaldutch.com/atmoscalc/index.htm
based on NASA-TM-X-74335.

Code obtained from github.com/OpenMDAO/dymos and modified to include option for deviation from standard day temperature.
A temperature correction is applied to the computed values, since the tables assume a fixed temperature for a given altitude.
The current setup does not account for attenuation of the temperature delta with increasing altitude (which ends at the 
tropopause), so this option must be toggled manually by the user to accurately represent real atmosphere conditions.
"""

import numpy as np
# import math

import openmdao.api as om
from atm1976data import atm_data as USatm1976
from MIL_SPEC_210A_tropical import atm_data as tropical_210A
from MIL_SPEC_210A_polar import atm_data as polar_210A
from MIL_SPEC_210A_hot import atm_data as hot_210A
from MIL_SPEC_210A_cold import atm_data as cold_210A


class USatmComp(om.ExplicitComponent):
    """
    Component model for the United States 
    - standard atmosphere 1976 tables
    - MIL-SPEC-201A atmosphere tables

    To build a set of custom atmosphere tables you will need to build new akima coefficients
    using _build_akima_coefs() at the end of this file and add your new coefficients as 
    an option intput to this component.

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

        self.options.declare('data_source', values=('USatm1976', 'tropical', 'polar', 'hot', 'cold'), default='USatm1976',
                             desc='The atmospheric model used, USATM1976 and MIL-SPEC-210A are available.')
        self.options.declare('h_def', values=('geopotential', 'geodetic'), default='geopotential',
                             desc='The definition of altitude provided as input to the component.  If "geodetic",'
                                  'it will be converted to geopotential based on Equation 19 in the original standard.')
        self.options.declare('output_dsos_dh', types=bool, default=False,
                             desc='If true, the derivative of the speed of sound will be added as an output')
        self.options.declare('output_abs_humidity', types=bool, default=False,
                             desc='If true, humidity derived from an empirical, non gradient-based model will be added as an output')
        self.options.declare('rel_humidity_sl', types=float, default=0.70,
                             desc='Fraction relative humidity at sea level to be used in absolute humidity model')
        self.options.declare('isa_delta_T', types=float, default=0.,
                             desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Rankine)')
        
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
        output_dsos_dh = self.options['output_dsos_dh']
        output_abs_humidity = self.options['output_abs_humidity']

        self._geodetic = self.options['h_def'] == 'geodetic'
        self._R0 = 6_356_766 / 0.3048  # Value of R0 from the original standard (m -> ft)

        self.add_input('h', val=1. * np.ones(nn), units='ft')

        self.add_output('temp', val=1. * np.ones(nn), units='degR')
        self.add_output('pres', val=1. * np.ones(nn), units='psi')
        self.add_output('rho', val=1. * np.ones(nn), units='slug/ft**3')
        self.add_output('viscosity', val=1. * np.ones(nn), units='lbf*s/ft**2')
        self.add_output('drhos_dh', val=1. * np.ones(nn), units='slug/ft**4')
        self.add_output('sos', val=1. * np.ones(nn), units='ft/s')
        if output_dsos_dh:
            self.add_output('dsos_dh', val=1. * np.ones(nn), units='1/s')
        if output_abs_humidity:
            self.add_output('abs_humidity', val=1. * np.ones(nn), units=None) # Given as % mole fraction

        arange = np.arange(nn, dtype=int)
        self.declare_partials(['temp', 'pres', 'rho', 'viscosity', 'drhos_dh', 'sos'], 'h',
                              rows=arange, cols=arange)
        if output_dsos_dh:
            self.declare_partials('dsos_dh', 'h', rows=arange, cols=arange)
        if output_abs_humidity:
            self.declare_partials('abs_humidity', 'h', rows=arange, cols=arange, method='cs')




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
        output_dsos_dh = self.options['output_dsos_dh']
        delta_T = self.options['isa_delta_T']
        output_abs_humidity = self.options['output_abs_humidity']
        rel_humidity_sl = self.options['rel_humidity_sl']

        if self._geodetic:
            h = h / (self._R0 + h) * self._R0  # Equation 19 from the original standard.

        # From this point forward, h is geopotential altitude (z in the original reference).

        idx = np.searchsorted(table_points, h, side='left')
        h_bin_left = np.hstack((table_points[0], table_points))
        dx = h - h_bin_left[idx]

        coeffs = self.source_data.akima_T[idx]
        T = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        outputs['temp'] = T + delta_T

        coeffs = self.source_data.akima_P[idx]
        pres = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        outputs['pres'] = pres #

        coeffs = self.source_data.akima_rho[idx]
        rho = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        outputs['rho'] = rho * T / (T + delta_T)
        drhos_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)
        outputs['drhos_dh'] = drhos_dh * T / (T + delta_T)

        coeffs = self.source_data.akima_viscosity[idx]
        viscosity = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
        outputs['viscosity'] = viscosity * ((T + 198.72)/(T + delta_T + 198.72)) * ((T + delta_T) / T)**1.5 # Sutherland's Law

        outputs['sos'] = np.sqrt(self._K * (T + delta_T))
        if output_dsos_dh:
            coeffs = self.source_data.akima_dT[idx]
            dT_dh = (coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))).ravel()
            outputs['dsos_dh'] = (0.5 / np.sqrt(self._K * (T + delta_T)) * dT_dh * self._K)
        
        if output_abs_humidity:
            idx = np.searchsorted(table_points, 0*h, side='left') # sea level
            h_bin_left = np.hstack((table_points[0], table_points))
            dx = 0 - h_bin_left[idx]

            coeffs = self.source_data.akima_T[idx]
            temp_SL = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
            coeffs = self.source_data.akima_P[idx]
            pres_SL = coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))
            TR = (T + delta_T) / temp_SL
            PR = pres / pres_SL
            abs_humidity = (rel_humidity_sl * 100 / PR) * 10**(8.4256 - (10.1995 / TR) - 4.922 * np.log(TR))
            outputs['abs_humidity'] = abs_humidity

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
        output_dsos_dh = self.options['output_dsos_dh']
        delta_T = self.options['isa_delta_T']
        output_abs_humidity = self.options['output_abs_humidity']
        rel_humidity_sl = self.options['rel_humidity_sl']

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
        dP_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx) #

        coeffs = self.source_data.akima_rho[idx]
        drho_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx) * T / (T + delta_T)

        coeffs = self.source_data.akima_viscosity[idx]
        dvisc_dh = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx) * ((T + 198.72)/(T + delta_T + 198.72)) * ((T + delta_T) / T)**1.5

        coeffs = self.source_data.akima_drho[idx]
        d2rho_dh2 = coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx) * T / (T + delta_T)
        
        partials['temp', 'h'][...] = dT_dh.ravel()
        partials['pres', 'h'][...] = dP_dh.ravel()
        partials['rho', 'h'][...] = drho_dh.ravel()
        partials['viscosity', 'h'][...] = dvisc_dh.ravel()
        partials['drhos_dh', 'h'][...] = d2rho_dh2.ravel()
        partials['sos', 'h'][...] = (0.5 / np.sqrt(self._K * (T + delta_T)) * partials['temp', 'h'] * self._K)
        if output_dsos_dh:
            coeffs = self.source_data.akima_dT[idx]
            _dT_dh = (coeffs[:, 0] + dx * (coeffs[:, 1] + dx * (coeffs[:, 2] + dx * coeffs[:, 3]))).ravel()
            d2T_dh2 = (coeffs[:, 1] + dx * (2.0 * coeffs[:, 2] + 3.0 * coeffs[:, 3] * dx)).ravel()
            partials['dsos_dh', 'h'] = 0.5 * np.sqrt(self._K / (T + delta_T)) * (d2T_dh2 - 0.5 * dT_dh**2 / (T + delta_T))

        if self._geodetic:
            partials['sos', 'h'][...] *= dz_dh
            partials['temp', 'h'][...] *= dz_dh
            partials['viscosity', 'h'][...] *= dz_dh
            partials['rho', 'h'][...] *= dz_dh
            partials['pres', 'h'][...] *= dz_dh
            partials['drhos_dh', 'h'][...] *= dz_dh ** 2
            if output_dsos_dh:
                partials['dsos_dh', 'h'] *= dz_dh ** 2





def _build_akima_coefs(raw_data, out_stream):
    """
    Print out the Akima coefficients based on the raw atmospheric data.

    This is used to more rapidly interpolate the data and the rate of change of rho wrt altitude.

    Returns
    -------
    dict
        A mapping of the variable name and Akima coeffcient values for each table in the atmosphere.
    """
    raw_data = np.reshape(raw_data, (raw_data.size // 6, 6))

    from collections import namedtuple
    atm_data = namedtuple('atm_data', ['alt', 'temp', 'pres', 'rho', 'a', 'viscosity'])

    # units='ft'
    atm_data.alt = raw_data[:, 0]

    # units='degR'
    atm_data.T = raw_data[:, 1]

    # units='psi'
    atm_data.P = raw_data[:, 2]

    # units='slug/ft**3'
    atm_data.rho = raw_data[:, 3]

    # units='ft/s'
    atm_data.a = raw_data[:, 4]

    # units='lbf*s/ft**2'
    atm_data.viscosity = raw_data[:, 5]

    import textwrap
    from openmdao.components.interp_util.interp import InterpND

    coeff_data = {}

    T_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.T, extrapolate=True)
    P_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.P, extrapolate=True)
    rho_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.rho, extrapolate=True)
    visc_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.viscosity,
                           extrapolate=True)

    _, _drho_dh = rho_interp.interpolate(atm_data.alt, compute_derivative=True)
    drho_interp = InterpND(method='1D-akima', points=atm_data.alt, values=_drho_dh.ravel(), extrapolate=True)

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
                print(f'atm_data.akima_{var} = \\', file=out_stream)
                print(textwrap.indent(repr(coeff_array).replace('array', 'np.array'), '    '),
                      file=out_stream)
                print('', file=out_stream)

            coeff_data[f'atm_data.akima_{var}'] = coeff_array

            input("Press Enter to continue: ")
    print("Program Complete")

    return coeff_data


if __name__ == "__main__":
    # Running this script generates and prints the Akima coefficients using the OpenMDAO akima1D interpolant.

    from USatmComp import _raw_data # replace this with your new raw data

    import sys
    _build_akima_coefs(raw_data=_raw_data, out_stream=sys.stdout)