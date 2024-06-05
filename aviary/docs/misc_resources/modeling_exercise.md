# Modeling Exercise for the Usability Study

This doc page details the modeling exercise for the usability study.
After attempting to complete the modeling exercises, please [fill out a survey to provide feedback on the usability of Aviary](https://forms.gle/pwNsRiwSWM5fwmTg9).
Please keep a rough estimation of the time it takes to complete each part of the exercise.

```{note}
Please give your best effort for completing each task, but do not worry if you are unable to complete the task. Your feedback is still valuable.
```

## Task 0: Installation

First, please install Aviary on your local system following the [installation instructions](../getting_started/installation.md).

## Task 1: Run a basic Aviary script

Aviary ships with multiple examples in the `aviary/examples` directory.
The most basic, `run_basic_aviary_example.py`, [(available here)](https://github.com/OpenMDAO/Aviary/blob/main/aviary/examples/run_basic_aviary_example.py) is our first starting point.
Please copy the contents of this file into a new Python script and run it from your terminal.

If the file runs successfully, you should see some output in your terminal that ends with:

```bash
Optimization terminated successfully    (Exit mode 0)
            Current function value: 2.420353717703553
            Iterations: 8
            Function evaluations: 8
            Gradient evaluations: 8
Optimization Complete
```

Once you have successfully run this script, please open the Aviary dashboard by running the following command in your terminal:

```bash
aviary dashboard run_basic_aviary_example
```

Check out the [dashboard docs](../user_guide/outputs_and_how_to_read_them.md) for more information on how to use the dashboard.
Play around with the dashboard and see what types of outputs are provided from Aviary.

## Task 2: Create a custom mission profile

The next step is to run an Aviary case with a mission profile that you define.
Please follow the [instructions in this example doc](../examples/simple_mission_example.ipynb) to create a custom mission profile.
Once you have created your custom mission profile, run the Aviary case following the instructions in the example doc.
If you are not able to successfully create a custom `phase_info` object to define the mission, please use the default one defined in the example.

```{note}
The survey will ask you to provide a copy of the `phase_info` object you create, so please save this information for later.
```

Again open the Aviary dashboard and visually examine the results by running the following command in your terminal:

```bash
aviary dashboard <name_of_the_script_you_ran (without .py)>
```

## Task 3: Run a mission with a reserve phase

Aviary has the capability to run a mission with a reserve phase as [detailed in the docs here](https://openmdao.github.io/Aviary/examples/reserve_missions.html).
The included example, `run_reserve_mission_fixedrange.py`, [(available here)](https://github.com/OpenMDAO/Aviary/blob/main/aviary/examples/reserve_missions/run_reserve_mission_fixedrange.py) demonstrates how to run a mission with a reserve phase.
Please copy the contents of this file into a new Python script.

Modify the target distance for the reserve phase to be 300 km and run the script.
Then open the Aviary dashboard to visually examine the results:

```bash
aviary dashboard run_reserve_mission_fixedrange
```

```{note}
Please record the value of the fuel burn for the `reserve_cruise` phase; the survey will ask you for this information.
```
