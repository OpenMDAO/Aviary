import textwrap

import numpy as np
from openmdao.components.interp_util.interp import InterpND

from aviary.variable_info.variables import Dynamic


def build_akima_coefs(out_stream, raw_data, units):
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

    atm_data = namedtuple(
        'atm_data',
        [
            Dynamic.Mission.ALTITUDE,
            Dynamic.Atmosphere.TEMPERATURE,
            Dynamic.Atmosphere.STATIC_PRESSURE,
            Dynamic.Atmosphere.DENSITY,
        ],
    )

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

    coeff_data = {}

    T_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.T, extrapolate=True)
    P_interp = InterpND(method='1D-akima', points=atm_data.alt, values=atm_data.P, extrapolate=True)
    rho_interp = InterpND(
        method='1D-akima', points=atm_data.alt, values=atm_data.rho, extrapolate=True
    )

    _, _dT_dh = T_interp.interpolate(atm_data.alt, compute_derivative=True)
    dT_interp = InterpND(
        method='1D-akima', points=atm_data.alt, values=_dT_dh.ravel(), extrapolate=True
    )

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
    build_akima = True

    if build_akima:
        ############### Generate Akima Splines Below ################
        # Running this script generates and prints the Akima coefficients using the OpenMDAO akima1D interpolant.

        print(
            'WARNING: build_akima_coefs() does not have the standard unit conversion capabilities '
            'you may be used to from OpenMDAO. Make sure your input units match the requirements '
            'shown in build_akima_coefs()!'
        )
        input('Press Enter to continue: ')

        from aviary.subsystems.atmosphere.data.MIL_SPEC_210A_Tropical import (
            _raw_data,
        )  # replace this with your new raw data

        _raw_data_units = 'English'  # replace this with your units ('SI' or 'English')

        import sys

        build_akima_coefs(out_stream=sys.stdout, raw_data=_raw_data, units=_raw_data_units)

    else:
        ################ Test problem below ################
        import openmdao.api as om

        from aviary.subsystems.atmosphere.atmosphere import AtmosphereComp
        from aviary.utils.aviary_values import AviaryValues
        from aviary.variable_info.enums import AtmosphereModel
        from aviary.variable_info.functions import setup_model_options
        from aviary.variable_info.variables import Settings

        prob = om.Problem()

        test_values = [-1000, 0, 10000, 35000, 55000, 70000, 100000]  # ft

        # 'standard', 'tropical', 'polar', 'hot', 'cold'
        atm_model = prob.model.add_subsystem(
            'comp',
            AtmosphereComp(delta_T_Celcius=0, num_nodes=len(test_values)),
            promotes=['*'],
        )

        options = AviaryValues()
        options.set_val(Settings.ATMOSPHERE_MODEL, val=AtmosphereModel.COLD)
        setup_model_options(prob, options)

        prob.set_solver_print(level=0)

        prob.setup(force_alloc_complex=True)

        prob.set_val('h', test_values, units='ft')

        prob.run_model()

        # prob.check_partials(method='cs')

        # print('Temperatures (K):', prob.get_val(Dynamic.Atmosphere.TEMPERATURE, units='K'))
        # print('Pressure (Pa)', prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, units='Pa'))
        # print('Density (kg/m**3)', prob.get_val(Dynamic.Atmosphere.DENSITY, units='kg/m**3'))
        # print('Viscosity (Pa*s)', prob.get_val(Dynamic.Atmosphere.DYNAMIC_VISCOSITY, units='Pa*s'))
        # print('Speed of Sound (m/s)', prob.get_val(Dynamic.Atmosphere.SPEED_OF_SOUND, units='m/s'))

        print('Temperatures (degF):', prob.get_val(Dynamic.Atmosphere.TEMPERATURE, units='degF'))
        print('Pressure (inHg60)', prob.get_val(Dynamic.Atmosphere.STATIC_PRESSURE, units='inHg60'))
        print('Density (lbm/ft**3)', prob.get_val(Dynamic.Atmosphere.DENSITY, units='lbm/ft**3'))
        print('Viscosity (Pa*s)', prob.get_val(Dynamic.Atmosphere.DYNAMIC_VISCOSITY, units='Pa*s'))
        print('Speed of Sound (m/s)', prob.get_val(Dynamic.Atmosphere.SPEED_OF_SOUND, units='m/s'))
