# Unit Tests

Tests help us know that Aviary is working properly as we update the code.
If you are adding to or modifying Aviary, you should add unit tests to ensure that your code is working as expected.

The goal of unit tests is to provide fine-grained details about how the functionality of the code is modified by your changes and to pinpoint any bugs that may have been introduced.
This page describes how to run the unit tests and how to add new tests.

## Running Unit Tests

Aviary ships with a suite of unit tests that can be run using the `testflo` command.
To run all unit tests, simply run `testflo` from the root directory of the Aviary repository.
To run a specific test, use the command `testflo <path_to_test_file>`.
We use the [testflo](https://github.com/naylor-b/testflo) package to run our unit tests as it allows us to run tests in parallel, which can significantly speed up the testing process.

Running `testflo .` at the base of the Aviary repository will run all unit tests and should produce output that looks like this (though the number of tests and skips may be different):

```bash
OK

Passed:  1065
Failed:  0
Skipped: 3


Ran 888 tests using 16 processes
Wall clock time:   00:00:54.15
```

## Writing Unit Tests
To write your own unittests, you should use the following utilities.

### assert_near_equal

The unit test that Aviary uses most is `assert_near_equal` from the OpenMDAO utility [assert_near_equal](https://openmdao.org/newdocs/versions/latest/_srcdocs/packages/utils/assert_utils.html). This assertion takes about 70% of all the assertions. It has the following format:

```
assert_near_equal(actual_value, expected_value, tolerance=1e-15, tol_type='rel')
```

where the `actual_value` is the value from Aviary and `expected_value` is what the developer expects. Ideally, the `expected_value` should come from computation by another tool (e.g. GASP, FLOPS or LEAPS1) or hand computation. When it is not possible, one can accept an Aviary computed value as expected. This guarantees that future development will not alter the outputs by mistake. As for the tolerance, it is good practice to take 1.e-6. By default, it checks relative error. If the `expected_value` is 0.0, it checks the absolute error.

One can find examples mostly in `subsystems` and `mission` The purpose is to make sure that a variable in an object (namely a component and/or a group) is computed as expected. It is advised that `assert_near_equal` test is carried for all outputs both in components and groups.

### assert_almost_equal and similar assertions

A similar unit test is NumPy's utility is `assert_almost_equal`. It checks whether the absolute difference of `actual_value` from `expected_value` is within certain tolerance. As [documented](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_almost_equal.html) by NumPy, it is not recommended.
For strings and integers, `assertEqual` of `unittest` is used. 

There are other similar assertions like `assertIsNone`, `assertIsNotNone`, `assertNotEqual`, `assertGreater`, `assertIn`, `assertNotIn`, `assertTrue`, and `assertFalse`. They together represent about 15% of all the assertions.


### assert_check_partials

The second most used assertion is `assert_check_partials` from the OpenMDAO utility. This is critically important because it checks whether the partial derivatives coded by develops are correct. It is the key in optimization. To use this test, you first prepare the partial derivative `data` by calling `check_partials` on a model after the model is setup. Then call `assert_check_partials` function with the `data`. Note that you don't need to run the model if you only want to check partials.

```
data = prob.check_partials(out_stream=None)
assert_check_partials(data, atol=1e-06, rtol=1e-06)
```

This assert makes sure that the computed derivatives in the code match those computed numerically within the given absolute and relative tolerances.

In Aviary, there are two ways to compute a component's derivatives: analytically or numerically. When the derivatives are analytic, it is best practice to use `check_partials` to compare them against the complex step (`cs`) or finite difference (`fd`) estimates.
Complex step is much more accurate, but all code in your component's `compute` method must be complex-safe to use this -- in other words, no calculation that squelches the imaginary part of the calculation (like `abs`.)
Note that there are some complex-safe alternatives to commonly-used calculations in the openmdao library. If your code is not complex-safe, or it wraps an external component that doesn't support complex numbers, then finite difference should be used.

If your component computes its derivatives numerically, there is less reason to test it because you are testing one numerical method against another.  If you choose to do this, you will need to use a different method, form, or step.
For example, if the partial derivatives are computed using `cs` method, you need to use `fd` method, or use `cs` method but with a different stepsize (e.g. `step=1.01e-40`):

```
data = prob.check_partials(out_stream=None, method="cs", step=1.01e-40)
assert_check_partials(data, atol=1e-06, rtol=1e-06)
```

Although the default method of `check_partials` is `fd` (finite difference), we prefer `cs` ([complex step](https://openmdao.org/newdocs/versions/latest/advanced_user_guide/complex_step.html) because it usually gives more accurate results.

````{margin}
```{note}
In general, we really don't have to check partials that are computed with complex step, since you expect that you should already be getting cs level of accuracy from them. Checks are primarily for analytic derivatives, where you can make mistakes. 
```
````

`check_partials` allows you to exclude some components in a group. For example `excludes=["*atmosphere*"]` means that atmosphere component will not be included.

Sometimes, you may need to exclude a particular partial derivative. You need to write your own code to do so. One example is in `subsystems/propulsion/test/test_propeller_performance.py`.

````{margin}
```{note}
Some of the partials in Aviary use table look up and interpolations. In the openmdao interpolation, the derivatives aren't always continuous if you interpolate right on one of the table points. You may need to tune the points you choose. For example, if 0.04 is a point in your test, you can change it to 0.04000001 and try again.
```
````

### assert_warning

This assertion checks that a warning is issued as expected. Currently, there is only one usage in Aviary but we should add more.

### assert_match_varnames

Another assertion used in tests is `assert_match_varnames` (about 4%). All of them are in `test_IO()` functions. It tests that all of the variables in an object (component or group) that are declared as inputs or outputs exist in the Aviary variable hierarchy. Exceptions are allowed by specifying `exclude_inputs` and `exclude_outputs`. For details, see [assert_utils.py](https://github.com/OpenMDAO/Aviary/blob/main/aviary/utils/test_utils/assert_utils.py).

### Other Assertions

Aviary has built several utility functions for unit tests:

- `assert_no_duplicates`
- `assert_structure_alphabetization`
- `assert_metadata_alphabetization`

## Adding Unit Tests

Whenever you add to Aviary, you should add a unit test to check its functionality.
Ideally, this test would check all logical paths in the code that you've added.
For example, if you add a new component, you should add a test that checks the component's output for a variety of inputs, as well as the partial derivative calculations.
If you add to an existing component, you should add a test that checks the new functionality you've added.

The logistical process of adding tests is relatively simple.
It's generally easier to look at existing tests to see how they're structured, but here's a quick overview.

Within the directory where you're adding or modifying code, there should be a `test` directory.
If there's not, you can create one.
Add a new file to this directory with the name `test_<name_of_file>.py` where `<name_of_file>` is the name of the file you're adding or modifying.
Within this file, add a class called `Test<name_of_file>` that inherits from `unittest.TestCase`.
Within this class, add a method called `test_<name_of_test>` where `<name_of_test>` is the name of the test you're adding.

Do not write docstrings for unittest methods, as they interfere with printouts while running testflo.
