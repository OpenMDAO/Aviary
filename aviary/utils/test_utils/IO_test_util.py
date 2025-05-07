from openmdao.utils.assert_utils import assert_near_equal


def check_prob_outputs(prob, vals, rtol=1e-6):
    """Check multiple problem outputs and print all failures.

    testflo doesn't handle unittest subTests, so this is a way to get all the
    failures at once without needing to iteratively fix and re-run. It also prints the
    variable path so you can more easily tell which assertions failed.

    Parameters
    ----------
    prob : Problem
        Problem instance to check. Should have already been run.
    vals : dict
        Dictionary mapping variable paths (i.e. ``prob[path]`` should work) to the
        value(s) to check against.
    rtol : float or list, optional
        Maximum relative tolerance to pass the test. If given as a list, it should match
        the number of entries in `vals`.
    """
    if isinstance(rtol, float):
        rtol = len(vals) * [rtol]

    errors = []
    for (path, val), t in zip(vals.items(), rtol):
        try:
            assert_near_equal(prob[path], val, tolerance=t)
        except ValueError as e:
            errors.append(f'\n  {path}: {e}')

    if errors:
        raise ValueError(''.join(errors))
