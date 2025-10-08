import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.testing_utils import use_tempdirs


from aviary.subsystems.aerodynamics.gasp_based.common import (
    AeroForces,
    CLFromLift,
    TanhRampComp,
    TimeRamp,
)
from aviary.variable_info.variables import Aircraft, Dynamic



import sys
import unittest
from itertools import chain

import numpy as np

from openmdao.utils.general_utils import add_border, get_max_widths, strs2row_iter
from openmdao.utils.om_warnings import reset_warning_registry, issue_warning


def assert_check_partials(data, atol=1e-6, rtol=1e-6, verbose=False, max_display_shape=(20, 20)):
    """
    Raise assertion if any entry from the return from check_partials is above a tolerance.

    Parameters
    ----------
    data : dict of dicts of dicts
            First key:
                is the component name;
            Second key:
                is the (output, input) tuple of strings;
            Third key:
                is one of ['tol violation', 'magnitude', 'J_fd', 'J_fwd', 'J_rev',
                           'vals_at_max_error', 'directional_fd_fwd',
                           'directional_fd_rev', 'directional_fwd_rev', 'rank_inconsistent',
                           'matrix_free', 'directional', 'steps', and 'rank_inconsistent'].

                For 'J_fd', 'J_fwd', 'J_rev' the value is a numpy array representing the computed
                Jacobian for the three different methods of computation.
                For 'tol violation' and 'vals_at_max_error' the value is a
                tuple containing values for forward - fd, reverse - fd, forward - reverse. For
                'magnitude' the value is a tuple indicating the maximum magnitude of values found in
                Jfwd, Jrev, and Jfd.
    atol : float
        Absolute error. Default is 1e-6.
    rtol : float
        Relative error. Default is 1e-6.
    verbose : bool
        When True, display more jacobian information.
    max_display_shape : tuple of int
        Maximum shape of the jacobians to display directly in the error message.
        Default is (20, 20).  Only active if verbose is True.
    """
    error_strings = []

    if isinstance(data, tuple):
        if len(data) != 2:
            raise RuntimeError(f"partials data format error (tuple of size {len(data)})")
        data = data[0]

    for comp in data:
        bad_derivs = []
        inconsistent_derivs = set()

        # Find all derivatives whose errors exceed tolerance.
        for key, pair_data in data[comp].items():
            if pair_data.get('rank_inconsistent'):
                inconsistent_derivs.add(key)

            J_fds = pair_data['J_fd']
            J_fwd = pair_data.get('J_fwd')
            J_rev = pair_data.get('J_rev')
            dir_fd_fwds = pair_data.get('directional_fd_fwd')
            dir_fd_revs = pair_data.get('directional_fd_rev')
            dir_fwd_rev = pair_data.get('directional_fwd_rev')
            directional = pair_data.get('directional')

            if not isinstance(J_fds, list):
                J_fds = [J_fds]
                dir_fd_fwds = [dir_fd_fwds]
                dir_fd_revs = [dir_fd_revs]
            else:
                if dir_fd_fwds is None:
                    dir_fd_fwds = [dir_fd_fwds]
                if dir_fd_revs is None:
                    dir_fd_revs = [dir_fd_revs]

            dirstr = ' directional' if directional else ''
            jacs = [(f'J_fwd{dirstr}', J_fwd, f'Forward{dirstr}'),
                    (f'J_rev{dirstr}', J_rev, f'Reverse{dirstr}')]

            steps = pair_data.get('steps', [None])

            nrows, ncols = J_fds[0].shape
            if isinstance(max_display_shape, int):
                maxrows = maxcols = max_display_shape
            else:
                try:
                    maxrows, maxcols = max_display_shape
                except ValueError:
                    issue_warning("max_display_shape must be an int or a tuple of two ints, but "
                                  f"got {max_display_shape}. Defaulting to (20, 20).")

            for J_fd, step, dfwd, drev in zip(J_fds, steps, dir_fd_fwds, dir_fd_revs):
                if step is not None:
                    stepstr = f" (step={step})"
                else:
                    stepstr = ""

                analytic_found = False
                for Jname, J, direction in jacs:
                    fwd = direction.startswith('Forward')
                    if J is not None:
                        analytic_found = True
                        try:
                            if fwd and dfwd is not None:
                                J1, J2 = dfwd
                                np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol,
                                                           verbose=False, equal_nan=False)
                            elif not fwd and drev is not None:
                                J1, J2 = drev
                                np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol,
                                                           verbose=False, equal_nan=False)
                            else:
                                J1, J2 = J, J_fd
                                np.testing.assert_allclose(J1, J2, atol=atol, rtol=rtol,
                                                           verbose=False, equal_nan=False)
                        except Exception as err:
                            abserr, relerr = _parse_assert_allclose_error(err.args[0])
                            if abserr < atol and not (np.any(J1) or np.any(J2)):
                                # if one array is all zeros and we don't violate the absolute
                                # tolerance, then don't flag the relative error.
                                continue

                            if verbose:
                                bad_derivs.append(f"\n{direction} derivatives of '{key[0]}' wrt "
                                                  f"'{key[1]}' do not match finite "
                                                  f"difference{stepstr}.\n")
                                bad_derivs[-1] += _filter_np_err(err.args[0])
                                if nrows <= maxrows and ncols <= maxcols:
                                    with np.printoptions(linewidth=10000):
                                        bad_derivs[-1] += f'\nJ_fd - {Jname}:\n' + \
                                            np.array2string(J_fd - J)
                            else:
                                bad_derivs.append([f"{key[0]} wrt {key[1]}", "abs",
                                                   f"fd-{Jname[2:]}", f"{abserr}"])
                                bad_derivs.append([f"{key[0]} wrt {key[1]}", "rel",
                                                   f"fd-{Jname[2:]}", f"{relerr}"])

                if not analytic_found:
                    # check if J_fd is all zeros.  If not, then we have a problem.
                    abserr = np.max(np.abs(J_fd))
                    if abserr > atol:
                        if verbose:
                            bad_derivs.append(f"\nAnalytic deriv for '{key[0]}' wrt '{key[1]}' "
                                              f"is assumed zero, but finite difference{stepstr} "
                                              "is nonzero.\n")
                            if nrows <= maxrows and ncols <= maxcols:
                                with np.printoptions(linewidth=10000):
                                    bad_derivs[-1] += '\nJ_fd - J_analytic:\n' + \
                                        np.array2string(J_fd)
                        else:
                            abserr = np.max(np.abs(J_fd))
                            bad_derivs.append([f"{key[0]} wrt {key[1]}", "abs", "fd-fwd",
                                               f"{abserr}"])

            if pair_data.get('matrix_free') is not None and J_fwd is not None and J_rev is not None:
                try:
                    if dir_fwd_rev is not None:
                        dJfwd, dJrev = dir_fwd_rev
                        either_zero = not (np.any(dJfwd) or np.any(dJrev))
                        np.testing.assert_allclose(dJfwd, dJrev, atol=atol, rtol=rtol,
                                                   verbose=False, equal_nan=False)
                    else:
                        either_zero = not (np.any(J_fwd) or np.any(J_rev))
                        np.testing.assert_allclose(J_fwd, J_rev, atol=atol, rtol=rtol,
                                                   verbose=False, equal_nan=False)
                except Exception as err:
                    abserr, relerr = _parse_assert_allclose_error(err.args[0])
                    if abserr < atol and either_zero:
                        # if one array is all zeros and we don't violate the absolute
                        # tolerance, then don't flag the relative error.
                        continue
                    if verbose:
                        bad_derivs.append(f"\nForward and Reverse derivatives of '{key[0]}' wrt "
                                          f"'{key[1]}' do not match.\n")
                        bad_derivs[-1] += _filter_np_err(err.args[0])
                        if nrows <= maxrows and ncols <= maxcols:
                            with np.printoptions(linewidth=10000):
                                bad_derivs[-1] += '\nJ_fwd - J_rev:\n' + \
                                    np.array2string(J_fwd - J_rev)
                    else:
                        bad_derivs.append([f"{key[0]} wrt {key[1]}", "abs", "fwd-rev", f"{abserr}"])
                        bad_derivs.append([f"{key[0]} wrt {key[1]}", "rel", "fwd-rev", f"{relerr}"])

        if bad_derivs or inconsistent_derivs:
            error_strings.extend(['', add_border(f'Component: {comp}', '-')])
            if bad_derivs:
                if verbose:
                    error_strings[-1] += '\n'.join(bad_derivs)
                else:
                    header = ['< output > wrt < variable >', 'max abs/rel', 'diff', 'value']
                    widths = get_max_widths(chain([header], bad_derivs))
                    header_str = list(strs2row_iter([header], widths, delim=' | '))[0]
                    error_strings.append(add_border(header_str, '-', above=False))
                    error_strings.extend(strs2row_iter(bad_derivs, widths, delim=' | '))

            if inconsistent_derivs:
                error_strings[-1] += (
                    "\nInconsistent derivs across processes for keys: "
                    f"{sorted(inconsistent_derivs)}.\nCheck that distributed outputs are properly "
                    "reduced when computing\nderivatives of serial inputs.")

    if error_strings:
        header = add_border('assert_check_partials failed for the following Components\n'
                            f'with absolute tolerance = {atol} and relative tolerance = {rtol}')
        err_string = '\n'.join(error_strings)
        raise ValueError(f"\n{header}\n{err_string}")


@use_tempdirs
class TestTanhRampComp(unittest.TestCase):
    def test_tanh_ramp_up(self):
        p = om.Problem()

        nn = 1000

        c = TanhRampComp(time_units='s', num_nodes=nn)

        c.add_ramp(
            'thruput',
            output_units='kg/s',
            initial_val=30,
            final_val=40,
            t_init_val=25,
            t_duration_val=5,
        )

        p.model.add_subsystem('tanh_ramp', c)

        p.setup(force_alloc_complex=True)

        p.set_val('tanh_ramp.time', val=np.linspace(0, 100, nn))

        p.run_model()

        cpd = p.check_partials(compact_print=True, method='cs', out_stream=None)

        thruput = p.get_val('tanh_ramp.thruput')

        import pprint
        pprint.pprint(cpd)

        assert_check_partials(cpd, atol=1.0e-9, rtol=1.0e-12)


if __name__ == '__main__':
    unittest.main()
