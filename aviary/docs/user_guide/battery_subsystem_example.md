# In-depth look at the battery subsystem as an example

## Examining the battery as an example case

Let's take a look at how to integrate an external subsystem into Aviary using the example of a battery model.
This battery model and builder are provided in the `aviary/examples/external_subsystems` folder.

We'll show you how to define the builder object, how to specify the pre-mission and mission models, and how to implement the interface methods.
In each of these methods, we've removed the initial docstring to increase readability on this page.
The [`battery_builder.py`](https://github.com/OpenMDAO/om-Aviary/blob/main/aviary/examples/external_subsystems/battery/battery_builder.py) file contains the original methods and docstrings.
Please see that file if you want to examine this example builder in its entirety as one file.

You can define these methods in any order you want within your subsystem builder and Aviary will use them at the appropriate times during analysis and optimization.
All of these methods must be defined by the subsystem builder creator, if that's you or one of your collaborators.

This battery example is just that -- a singular example of the types of external subsystems that Aviary can include.
We don't expect the full complexity of all disciplines' systems to be reflected in this example, so please keep that in mind when viewing these example methods.

## Defining the builder object

The first step is to define a builder object for your external subsystem.
This class extends the `SubsystemBuilderBase` class and provides implementations for all of the interface methods that Aviary requires.

```python
class BatteryBuilder(SubsystemBuilderBase):
    def __init__(self, name='battery', include_constraints=True):
        self.include_constraints = include_constraints
        super().__init__(name, meta_data=ExtendedMetaData)
```

## Defining the variables of interest: states, controls, constraints, etc

First, we'll cover how to tell Aviary which variables are states that you want to integrate throughout the mission.
The battery subsystem has two states: SOC (state of charge) and Thevenin voltage.
Both of these states are described below using their variable hierarchy name, then a dictionary that tells Aviary any arguments that are passed to the Dymos `add_state` command.
You can use any arguments that the `add_state` command accepts here, including `ref`s, `bound`s, `solve_segments`, and more.
Please see the [Dymos docs for states](https://openmdao.github.io/dymos/features/phases/variables.html#states) for a table of all the available input args.

```python
def get_states(self):
    states_dict = {
        Mission.Battery.STATE_OF_CHARGE: {
            'rate_source': Mission.Battery.STATE_OF_CHARGE_RATE,
            'fix_initial': True,
        },
        Mission.Battery.VOLTAGE_THEVENIN: {
            'units': 'V',
            'rate_source': Mission.Battery.VOLTAGE_THEVENIN_RATE,
            'defect_ref': 1.e5,
            'ref': 1.e5,
        },
    }

    return states_dict
```

We have similar methods for each of constraints, parameters, and design variables.

For the constraints, you provide the name of the variable you want to constrain as the key in a dictionary.
Then the values for each one of these keys is a dictionary of kwargs that can be passed to the [Dymos `add_constraint()` method](https://openmdao.github.io/dymos/features/phases/constraints.html#constraints).
The example for the battery shows that we have a final boundary constraint on the state of charge being above 0.2 as well as a path constraint on the Thevenin voltage.

```{note}
The constraint dicts used in Aviary require one additional kwarg on top of those needed by Dymos: the `type` argument. You must specify your constraint as either a `boundary` or `path` constraint.
```

```python
def get_constraints(self):
    if self.include_constraints:
        constraints = {
            Mission.Battery.STATE_OF_CHARGE: {
                'lower': 0.2,
                'type': 'boundary',
                'loc': 'final',
            },
            Mission.Battery.VOLTAGE_THEVENIN: {
                'lower': 0,
                'type': 'path',
            },
        }
    else:
        constraints = {}

    return constraints
```

Next up we have any parameters added to the phase from the external subsystem.
[Parameters in the Dymos sense](https://openmdao.github.io/dymos/features/phases/variables.html#parameters) are non-time-varying inputs since they typically involve fixed-value variables within a phase which define a system.
These are fixed across a phase or trajectory, but can be controlled by the optimizer.
For the simple battery case here we're setting the operating temperature of the battery to be fixed at a certain value.

```python
def get_parameters(self):
    parameters_dict = {
        Mission.Battery.TEMPERATURE: {'val': 25.0, 'units': 'degC'},
        Mission.Battery.CURRENT: {'val': 3.25, 'units': 'A'}
    }

    return parameters_dict
```

If you want to add high-level design variables to your external subsystem that are not exposed at the phase level, you can do so by creating a method that describes your desired design variables.
This method is called `get_design_vars`.
There might be calculations at the pre-mission level that necessitate using this method to allow the optimizer to control high-level sizing variables.
In the simple battery case, we allow the optimizer to control the energy capacity of the battery, effectively sizing it.

```python
def get_design_vars(self):
    DVs = {
        Aircraft.Battery.Cell.DISCHARGE_RATE: {
            'units': 'A',
            'lower': 0.0,
            'upper': 1.0,
        },
    }

    return DVs
```

## Defining the OpenMDAO systems needed

Next up, we have the two methods for getting the mission and pre-mission models, `get_mission` and `get_pre_mission`, respectively.
When you define these methods they should return OpenMDAO Systems ([groups](https://openmdao.org/newdocs/versions/latest/_srcdocs/packages/core/group.html) or [components](https://openmdao.org/newdocs/versions/latest/_srcdocs/packages/core/component.html)) that represent the mission and pre-mission subsystems of the external model.
Please see the [OpenMDAO docs on creating a simple component](https://openmdao.org/newdocs/versions/latest/basic_user_guide/single_disciplinary_optimization/first_analysis.html) for more information on how to create OpenMDAO systems.
In short, an OpenMDAO System takes in input variables, does some calculations, and returns output variables.
If you have an analysis or design code, you can wrap it in an OpenMDAO System even if it's not Python-based; [this doc page](https://openmdao.org/newdocs/versions/latest/other_useful_docs/file_wrap.html) goes into more detail about that. 

In the case of the battery model, the pre-mission System is very simple -- it's just a single component with no special setup or arguments.
As such, the method definition looks like this:

```python
def build_pre_mission(self):
    return BatteryPreMission()
```

That's pretty straightforward.
The `build_pre_mission` method requires the `aviary_inputs` object (an instantiation of an `AviaryValues` object).
This battery example does not use any information from `aviary_inputs`.
The pre-mission builder can then use the data and options within the `aviary_inputs` object to construct the OpenMDAO System using user-specified logic.

Now we'll discuss arguably the most important method needed when making external subsystems (though they're all important): `build_mission`.
This method returns an OpenMDAO System that provides all computations needed during the mission.
This includes computing state rates so that the mission integration code can compute the state values and obtain the corresponding performance of the aircraft.

For the battery case, we have a nicely packaged `BatteryMission` System, so our method looks like this:

```python
def build_mission(self, num_nodes, aviary_inputs):
    return BatteryMission(num_nodes=num_nodes, aviary_inputs=aviary_inputs)
```

Note that `build_mission` requires both `num_nodes` and `aviary_inputs` as arguments.
`num_nodes` is needed to correctly set up all the vectors within your mission subsystem, whereas `aviary_inputs` helps provide necessary options for the subsystem.

## Defining helper methods, like preprocessing, initial guessing, and linking

Next up we have some helper methods that allow us to preprocess the user-provided inputs, link different phases' variables, and provide initial guesses.

Let's talk about preprocessing your inputs.
You might want to have some logic that sets different values in your problem based on user-provided options.
`preprocess_inputs` allows you to do this.
It occurs between `load_inputs` and `build_pre_mission` in the Aviary stack, so after the initial data inputs are loaded but before they're used to instantiate any OpenMDAO models.
This is a great place to set any values you need for variables within your subsystem.

```{note}
For both users and developers: `preprocess_inputs` happens **once** per analysis or optimization run, just like loading the inputs. It does not occur as part of an OpenMDAO system, so it does not get iterated over during the optimization process.
```

You can have calculations or checks you need in this method based on user inputs.

Now you may want to link some variables between phases.
States, for example, are usually great candidates for linking.
In the case of the battery example, if we have a climb and then a cruise phase, we'd want to connect the state-of-charge and voltage states so the end of climb is equal to the beginning of cruise.
Thus, our `get_linked_variables` method looks like this:

```python
def get_linked_variables(self):
    '''
    Return the list of linked variables for the battery subsystem; in this case
    it's our two state variables.
    '''
    return [Mission.Battery.VOLTAGE_THEVENIN, Mission.Battery.STATE_OF_CHARGE]
```

The last method we'll discuss here is `get_initial_guesses`.
This method allows you to return info for any variable you want to provide an initial guess for.
Depending on the integration method you're using, initial guesses may greatly help convergence of your optimization problem.
Collocation methods especially benefit from good initial guesses.

For this battery example we simply define some initial guesses for the state variables.
If we wanted to give other initial guesses we could specify their `type` in the dictionary accordingly.
These values apply to all nodes across the mission, e.g. each node gets the same value guess. 

```python
def get_initial_guesses(self):
    initial_guess_dict = {
        Mission.Battery.STATE_OF_CHARGE: {
            'val': 1.0,
            'type': 'state',
        },
        Mission.Battery.VOLTAGE_THEVENIN: {
            'val': 5.0,
            'units': 'V',
            'type': 'state',
        },
    }

    return initial_guess_dict
```

## Extending the variable hierarchy

You might've noticed throughout this battery example that we've extended the core variable hierarchy included in Aviary to add variables needed for the battery system.
To handle additional variables not already defined in Aviary, you can define an extension to the available variables.
By following the same naming convention present through Aviary, specifically that variables start with `aircraft` and `mission`, these can be correctly handled by the Aviary tool.

<!-- TODO: This section needs much more verbosity and detail! -->

## Testing your implementation

Once you have written your external subsystem code and built your own subsystem class, you can test your implementation using a built-in tool.
We provide a set of unit tests that accept a user-defined subsystem and check that the outputs from the methods match what Aviary expects.
If the user-defined methods do not return the correct form of the data, the test raises appropriate exceptions and explains which arguments are missing.

By running these tests on their own subsystem, users can ensure that their code meets the requirements and standards set by the aviary package.
Specifically, the tests can help users check that their subsystem's inputs and outputs are correct and consistent with the expected format, even before running an Aviary mission.

This test works by inheriting from a base class that loops through each of the methods and tests the outputs.
All you'd have to do is provide your builder object as well as the `aviary_inputs` object needed to for the methods in the builder object.

Here's an example of the full code that you would write to test the battery builder.
Although there are no unit tests explicitly shown in this file, they are contained in the `TestSubsystemBuilderBase` class, so you only need these few lines to test your subsystem.

```python
from aviary_examples.subsystems.battery.battery_builder import BatteryBuilder
from aviary_examples.subsystems.battery.battery_variables import Aircraft, Mission

import aviary.api as av

class TestBattery(av.TestSubsystemBuilderBase):

    def setUp(self):
        self.subsystem_builder = BatteryBuilder()
        self.aviary_inputs = av.AviaryValues()
```

For instance, if you saved this class in a file called `test_battery.py`, you could then run `testflo test_battery.py` to verify that all the methods do what Aviary expects.
If everything is good with your model you'll see output in your terminal like this:

```bash
(dev) $ testflo test_battery.py
......................................

OK

Passed:  38
Failed:  0
Skipped: 0


Ran 38 tests using 16 processes
Wall clock time:   00:00:4.98
```

If something's wrong with your builder, this test should tell you which method is out of spec and how you can fix it.
For example, here I've purposefully made the battery builder have some incorrect behavior and reran the test.
Here's the output:

```bash
(dev) $ testflo test_battery.py
...............test_battery.py:TestBattery.test_get_states ... FAIL (00:00:0.01, 139 MB)
Traceback (most recent call last):
  File "/home/user/anaconda3/envs/dev/lib/python3.8/site-packages/testflo/test.py", line 418, in _try_call
    func()
  File "/mnt/c/Users/user/git/Aviary/aviary/subsystems/test/subsystem_tester.py", line 18, in test_get_states
    self.assertIsInstance(
  File "/home/user/anaconda3/envs/dev/lib/python3.8/unittest/case.py", line 1335, in assertIsInstance
    self.fail(self._formatMessage(msg, standardMsg))
  File "/home/user/anaconda3/envs/dev/lib/python3.8/unittest/case.py", line 753, in fail
    raise self.failureException(msg)
AssertionError: 6 is not an instance of <class 'dict'> : the value for 'amps' should be a dictionary

test_battery.py:TestBattery.test_get_constraints ... FAIL (00:00:0.01, 139 MB)
Traceback (most recent call last):
  File "/home/user/anaconda3/envs/dev/lib/python3.8/site-packages/testflo/test.py", line 418, in _try_call
    func()
  File "/mnt/c/Users/user/git/Aviary/aviary/subsystems/test/subsystem_tester.py", line 56, in test_get_constraints
    self.assertIn('type', values.keys(),
  File "/home/user/anaconda3/envs/dev/lib/python3.8/unittest/case.py", line 1179, in assertIn
    self.fail(self._formatMessage(msg, standardMsg))
  File "/home/user/anaconda3/envs/dev/lib/python3.8/unittest/case.py", line 753, in fail
    raise self.failureException(msg)
AssertionError: 'type' not found in dict_keys(['lower', 'upper']) : Constraint "amps" is missing the "type" key

...

The following tests failed:
test_battery.py:TestBattery.test_get_constraints
test_battery.py:TestBattery.test_get_states


Passed:  18
Failed:  2
Skipped: 0


Ran 20 tests using 16 processes
Wall clock time:   00:00:6.87
```

The output is a bit verbose, but tells you which methods are incorrect and why.
For example, here the `get_states` method returned a dict that included a key (`'amps'`) with a value of 6 instead of being a dictionary as expected.
In the `get_constraints` method, a constraint was added to the dictionary but did not include a `type` key, which is required as stated by the error message.

If you encounter an error when using your subsystem, but the test here did not find it, please let the Aviary dev team know!
We'd love to hear from you on the [GitHub issues page](https://github.com/OpenMDAO/om-Aviary/issues) so we can help everyone write great external subsystems.