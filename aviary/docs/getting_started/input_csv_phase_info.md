# Vehicle Input .csv File and Phase_info Dictionary

## Vehicle Input .csv File

```{note}
This section is under development.
```

`initial_guesses` is a dictionary that contains values used to initialize the trajectory. It contains the following keys along with default values:

- actual_takeoff_mass: 0,
- rotation_mass: .99,
- operating_empty_mass: 0,
- fuel_burn_per_passenger_mile: 0.1,
- cruise_mass_final: 0,
- flight_duration: 0,
- time_to_climb: 0,
- climb_range: 0,
- reserves: 0

The initial guess of `reserves` is used to define the reserve fuel. Initially, its value can be anything larger than or equal to 0. There are two Aviary variables to control the reserve fuel in the model file (`.csv`):
- `Aircraft.Design.RESERVE_FUEL_ADDITIONAL`: the required fuel reserves: directly in lbm,
- `Aircraft.Design.RESERVE_FUEL_FRACTION`: the required fuel reserves: given as a proportion of mission fuel.

If the value of initial guess of `reserves` (also in the model file if any) is 0, the initial guess of reserve fuel comes from the above two Aviary variables.  Otherwise, it is determined by the parameter `reserves`:
- if `reserves > 10`, we assume it is the actual fuel reserves.
- if `0.0 <= reserves <= 10`, we assume it is the fraction of the mission fuel.

The initial guess of `reserves` is always converted to the actual design reserves (instead of reserve factor) and is used to update the initial guesses of `fuel_burn_per_passenger_mile` and `cruise_mass_final`.

## Phase Info Dictionary

`phase_info` is a nested dictionary that Aviary uses for users to define their mission phases - how they are built, the design variables, constraints, connections, etc.

We will now discuss the meaning of the keys within the `phase_info` objects.

- If a key starts with `min_` or `max_` or ends with `_lower` or `_upper`, it is a lower or upper bound of a state variable. The following keys are not state variables:
  - `required_available_climb_rate`: the minimum rate of climb required from the aircraft at the top of climb (beginning of cruise) point in the mission. You don't want your available rate-of-climb to be 0 in case you need to gain altitude during cruise.
  - `EAS_limit`: the maximum descending EAS in knots.
  - `throttle_setting`: the prescribed throttle setting. This is only used for `GASP` and `solved` missions.
- If a key ends with `_ref` or `_ref0` (except `duration_ref`, `duration_ref0`, `initial_ref` and `initial_ref0`), it is the unit-reference and zero-reference values of the control variable at the nodes. This option is invalid if opt=False. Note that it is a simple usage of  ref and ref0. We refer to [Dymos](https://openmdao.github.io/dymos/api/phase_api.html?highlight=ref0#add-state) for details.
- Some keys are for phase time only.
  - `duration_ref` and `duration_ref0` are unit-reference and zero reference for phase time duration.
  - `duration_bounds` are the bounds (lower, upper) for the time duration of the phase.
  - `initial_ref` and `initial_ref0` are the unit-reference and zero references for the initial value of time.
  - `time_initial_ref` and `time_initial_ref0` are the unit-reference and zero-reference for the initial value of time.
  - `initial_bounds`: the lower and upper bounds of initial time. For `GASP`, it is `time_initial_bounds`.
- If a key starts with `final_`, it is the final value of a state variable.
- If a key ends with `_constraint_eq`, it is an equality constraint.

- Keys related to altitude:
  - We use `final_altitude` to indicate the final altitude of the phase.
  - Meanwhile, `alt` is a key in acceleration phase parameter for altitude in `GASP` missions and `altitude` is a key in all other phases of all missions.

- Some keys are a boolean flag of True or False:
  - `input_initial`: the flag to indicate whether initial values of of a state (such as: altitude, velocity, mass, etc.) is taken.
  - `add_initial_mass_constraint`: the flag to indicate whether to add initial mass constraint
  - `clean`: the flag to indicate no flaps or gear are included.
  - `connect_initial_mass`: the flag to indicate whether the initial mass is the same as the final mass of previous phase.
  - `fix_initial`: the flag to indicate whether the initial state variables is fixed.
  - `fix_initial_time`: the flag to indicate whether the initial time is fixed.
  - `no_climb`: if True for the descent phase, the aircraft is not allowed to climb during the descent phase.
  - `no_descent`: if True for the climb phase, the aircraft is not allowed to descend during the climb phase.
  - `include_landing`: the flag to indicate whether there is a landing phase.
  - `include_takeoff`: the flag to indicate whether there is a takeoff phase.
  - `optimize_mass`: if True, the gross takeoff mass of the aircraft is a design variable.
  - `target_mach`: the flag to indicate whether to target mach number.
- `initial_guesses`: initial guesses of state variables.
- `COLLOCATION` related keys:
  - `num_segments`: the number of segments in transcription creation in Dymos. The minimum value is 1. This is needed if 'AnalysisScheme' is `COLLOCATION`.
  - `order`: the order of polynomials for interpolation in transcription creation in Dymos. The minimum value is 3. This is needed if 'AnalysisScheme' is `COLLOCATION`.
- Other Aviary keys:
  - `subsystem_options`: The `aerodynamics` key allows two methods: `computed` and `solved_alpha`. In case of `solved_alpha`, it requires an additional key `aero_data_file`.
  - `external_subsystems`: a list of external subsystems.
- other keys that are self-explanatory:
  - `clean`: a flag for low speed aero (which includes high-lift devices) or cruise aero (clean, because it does not include high-lift devices).
  - `EAS_target`: the target equivalent airspeed.
  - `initial_mach`: initial mach number.
  - `linear_solver`:  provide an instance of a [LinearSolver](https://openmdao.org/newdocs/versions/latest/features/core_features/controlling_solver_behavior/set_solvers.html) to the phase.
  - `mach_cruise`: the cruise mach number.
  - `mass_f_cruise`: final cruise mass (kg). It is used as `ref` and `defect_ref` in cruise phase.
  - `nonlinear_solver`: provide an instance of a [NonlinearSolver](https://openmdao.org/newdocs/versions/latest/features/core_features/controlling_solver_behavior/set_solvers.html) to the phase.
  - `ode_class`: default to `MissionODE`.
  - `range_f_cruise`: final cruise range (m). It is used as `ref` and `defect_ref` in cruise phase.
  - `solve_segments`: False, 'forward', 'backward'. This is a Radau option.
  - `polynomial_control_order`: default to `None`.
  - `use_actual_takeoff_mass`: default to `False`.
  - `fix_duration`: default to `False`.
  - `solve_for_distance`: if True, use a nonlinear solver to converge the `distance` state variable to the desired value. Otherwise use the optimizer to converge the `distance` state.
  - `optimize_mach`: if True, the Mach number is a design variable.
  - `optimize_altitude`: if True, the altitude is a design variable.
  - `constraints`: a dictionary of user-defined constraints. The keys are the names of the constraints and the values are the keyword arguments expected by Dymos.

```{note}
Not all the keys apply to all phases. The users should select the right keys for each phase of interest. The required keys for each phase are defined in [check_phase_info](https://github.com/OpenMDAO/Aviary/blob/main/aviary/interface/utils.py) function. Currently, this function does the check only for `FLOPS` and `GASP` missions.
```

## Using custom phase builders

For the `height_energy`, you can use a user-defined phase builder.
The user-defined phase builder must inherit from `PhaseBuilderBase` and provide the `from_phase_info` and the `build_phase` methods.
The `from_phase_info` method is used to convert the `phase_info` dictionary into the inputs needed for the phase builder object.
The `build_phase` method is used to actually build and output the `Phase` object.

For examples of how to create a custom phase builder, see the `energy_phase.py` file.

```{note}
Using custom phase builders is a particularly advanced feature and is not recommended for most users.
```
