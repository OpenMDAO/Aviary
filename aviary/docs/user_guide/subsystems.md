# SubsystemBuilderBase

## Method Overview

Here is a brief overview of the available methods that are used in the `SubsystemBuilderBase` object.
The docstrings within this builder base class go into much more detail.
This overview is automatically generated from the docstrings in the builder base class.

We'll now detail where in the Aviary stack each one of these methods is used.
Understanding this can be helpful for knowing which parts of the Aviary problem will be impacted by your subsystem.
In the following outline, the methods listed at the top-level are defined in `methods_for_level2.py` and are called in this order to run an Aviary problem.
Any sub-listed method is one that you can provide with your subsystem builder, showing where within the level 3 method hierarchy that subsystem method gets used.

- `load_inputs` - loads the aviary_values inputs and options that the user specifies.
- `check_and_preprocess_inputs` - checks the user-supplied input values for any potential problems.
  - `preprocess_inputs`
- `add_pre_mission_systems` - adds pre-mission Systems to the Aviary problem
  - `get_mass_names`
  - `build_pre_mission`
- `add_phases` - adds mission phases to the Aviary problem
  - `get_states`
  - `get_constraints`
  - `get_controls`
  - `get_parameters`
  - `build_mission`
- `add_post_mission_systems` - adds the post-mission Systems to the Aviary problem
  - `build_post_mission`
- `link_phases` - links variables between phases
  - `get_linked_variables`
  - `get_bus_variables`
- `add_driver` - adds the driver (usually an optimizer)
- `add_design_variables` - adds the optimization design variables
  - `get_design_vars`
- `add_objective` - adds the user-selected objective
- `setup` - sets up the Aviary problem
  - `get_outputs`
  - `define_order`
- `set_initial_guesses` - sets the initial guesses for the Aviary problem
  - `get_initial_guesses`
- `run_aviary_problem` - actually runs the Aviary problem

```{note}
Understanding the flow of the above methods and how the subsystem methods are used within Aviary is pretty important! Make sure to review these methods and where in the stack they're used before digging too deep into debugging your subsystem.
```
