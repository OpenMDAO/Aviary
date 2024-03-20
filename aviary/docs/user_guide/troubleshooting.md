# Troubleshooting

## Building Understanding

A fantastic feature of Aviary is the fact that you can construct ridiculously complex coupled aircraft-mission design problems that span multiple disciplines and fidelities.
However, this can also be a curse.
When you're first starting out with Aviary, it can be difficult to understand why your model is behaving the way it is.
This section will discuss some strategies for building up an understanding of your model and how to troubleshoot when things aren't working as expected.

````{margin}
```{note}
It'd be wonderful if optimization allowed you to press one button and create the best aircraft. Unfortunately, that's not the case. We still need engineering intuition and understanding to build good models and interpret results.
```
````

A valuable resource that we've already developed is the [Debugging your optimizations](https://openmdao.github.io/PracticalMDO/Notebooks/Optimization/debugging_your_optimizations.html) content from the [Practical Multidisciplinary Design Optimization](https://openmdao.github.io/PracticalMDO/) course.
This video and notebook discuss how to build up an understanding of your optimization model in a general sense.

This doc page will discuss how to build up an understanding of your Aviary model in particular.

### Understand your subsystem models

The first step in building up an understanding of your complete model is to understand the subsystem models you're using.
For example, if you're using an engine deck for the propulsion model, plotting the thrust and fuel flow as a function of Mach number and altitude can be a great way to understand how the engine will behave.
Similarly, if you're using a battery model, plotting the battery state of charge as a function of time with the expected power draw can be a great way to understand how the battery will behave.

Without a thorough understanding of your subsystem models, it will be nearly impossible to understand how and why the optimizer is making certain decisions.
Famously, optimizers are very good at finding parts of the model space that are poorly defined and exploiting them in pursuit of minimizing the objective.

### Start with a simple mission

The first step in building up an understanding of your model is to start simple.
This might sound straightforward, but you should start with a simple aircraft model and a simple mission.
For example, if you want to eventually model a hybrid-electric aircraft flying a fully optimized trajectory, you might want to start with a simpler mission where the climb rate and cruise altitude are fixed.
Once you get good results with the simple mission and understand the results, you can start adding complexity and flexibility.

### Interpreting optimized results

```{note}
A `VERBOSITY` control has been added to minimize the amount of unnecessary information that will be displayed.
Currently Quiet, Brief [default], Verbose, and Debug are supported. Quiet will suppress practically everything other than warnings and errors. Verbose will include information such as the progress of the optimization, instead of just a final summary. And Debug will contain detailed information about many of the steps as they happen.
Some of Aviary's CLI functions, such as `fortran_to_aviary`, allow the verbosity to be set directly with a command line argument. `run_mission` uses the variable `settings:verbosity` to control the print levels.
```

Once you've built up an understanding of your model and have successfully performed optimization, you can start to interpret the results.
This is where a mix of aircraft engineering knowledge and optimization knowledge is extremely helpful.

First, examine the exit code of the optimizer.
If the optimizer exited with a non-zero exit code, it means that the optimizer did not converge well to a solution.
This could be due to a number of reasons, such as the prescribed constraints being unsatisfiable or that the optimizer had numerical difficulties finding the optimum.

In the event of non-convergence, you should see if you are solving the simplest relevant optimization case for your studies.
If you aren't, it's beneficial to start with the simplest case and build up complexity until you find the source of the non-convergence.

If the optimizer exited with a zero exit code, it means that the optimizer converged to a solution.
Now you can start to interpret the resulting trajectory and aircraft design.

Aviary provides a number of reports to help you interpret the results, including the `opt_report.html` and `traj_results_report.html`.

The `opt_report.html` shows you the final values of the design variables, constraints, and objective, along with the corresponding bounds for each value.
This is helpful in determining which design variables are at their limits as well as which constraints are driving the design of the aircraft.

The `traj_results_report.html` shows you plots of the trajectory variables as a function of time.
Specifically, you can look at the altitude and Mach profiles to see if the aircraft flight is in line with what you expected.
You can also view any tracked state variables, such as the battery state of charge, to see if the subsystems are behaving as expected.

## Ensuring Subsystems Compatibility

This section is under development.
