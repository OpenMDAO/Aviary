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

Passed:  885
Failed:  0
Skipped: 3


Ran 888 tests using 16 processes
Wall clock time:   00:00:54.15
```

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
