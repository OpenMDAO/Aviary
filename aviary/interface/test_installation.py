"""
NOTE this file cannot be a unittest, as testflo is an optional dependency. Instead, a separate
     unittest that imports and runs this function exists under interface/tests/test_installation.py.
"""


def _setup_installation_test(parser):
    """There are no arguments for `aviary check`."""
    pass


def _exec_installation_test(args, user_args):
    """
    Tests your Aviary installation by importing the API, and then running an example case.
    Printouts are provided to explain what was run in the case that there are multiple options, and
    a confirmation that the entire run was error free.

    Returns
    -------
    success : bool
        State of installation tests: True = successful, False = problems with installation
    """
    success = True
    print('Testing imports:')
    # Test that openmdao can be imported
    print('Importing OpenMDAO')
    try:
        from openmdao.utils.testing_utils import use_tempdirs
    except ImportError:
        print('ERROR: OpenMDAO is not installed')
        return False
    else:
        print('success')

    # Try importing the Aviary api - there are many files imported here so a large number of errors
    # are possible. Catch any exception and show to user.
    print('Importing Aviary api')
    try:
        import aviary.api as av
    except Exception as import_error:
        print(f'An error occurred while importing Aviary API: {import_error}\n')
        return False
    else:
        print('success')

    @use_tempdirs
    def _test_install(optimizer):
        """Runs an example Aviary problem using the requested optimizers in a temporary directory."""
        return av.run_aviary(
            'models/test_aircraft/aircraft_for_bench_FwFm.csv',
            av.default_height_energy_phase_info,
            optimizer=optimizer,
            make_plots=False,
            verbosity=0,
        )

    # Check for pyoptsparse, let user know if it is found or not
    print('Importing pyOptSparse')
    try:
        from pyoptsparse import OPT
    except ImportError:
        print('pyOptSparse is not installed')
        use_pyoptsparse = False
    else:
        print('success')
        use_pyoptsparse = True

    # Check for which optimizers are available
    # SLSQP is default
    optimizers = ['SLSQP']
    if use_pyoptsparse:
        # First test the IPOPT can be found
        try:
            OPT('IPOPT')
        except Exception:
            # IPOPT is optional, so it isn't an issue if we don't find it
            pass
        else:
            optimizers.append('IPOPT')
        # Next test if SNOPT is available. Use it if so.
        try:
            OPT('SNOPT')
        except Exception:
            # SNOPT is optional, so it isn't an issue if we don't find it
            pass
        else:
            optimizers.append('SNOPT')

    # Tell user which optimizers are available
    print(f'The following optimizers are available for use: {optimizers}')
    optimizer = optimizers[-1]
    print(f'\nRunning a basic Aviary model using the {optimizer} optimizer:')

    try:
        prob = _test_install(optimizer)
    except Exception as error:
        print(f'The following error occurred while running the Aviary problem: {error}')
        success = False
    else:
        if prob.problem_ran_successfully:
            print('Aviary run successful')
            print(
                '\nYour Aviary installation is working. Please review the printouts to make sure '
                'all expected features are present.'
            )
        else:
            print('Aviary run failed')
            print(
                '\nYour Aviary installation is not working properly. Please review the printouts '
                'to determine which step failed.'
            )
            success = False

    return success


if __name__ == '__main__':
    _exec_installation_test()
