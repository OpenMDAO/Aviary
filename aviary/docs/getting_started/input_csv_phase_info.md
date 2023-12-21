# Vehicle Input .csv File and Phase_info Dictionary

## Vehicle Input .csv File

```{note}
This section is under development.
```

## Phase Info Dictionary

`phase_info` is a nested dictionary that Aviary uses for users to define their mission phases - how they are built, the design variables, constraints, connections, etc.

We will now discuss the meaning of the keys within the `phase_info` objects.

- If a key starts with `min_` or `max_` or ends with `_lower` or `_upper`, it is a lower or upper bound of a state variable. The following keys are not state variables:
  - `required_available_climb_rate`: the minimum rate of climb required from the aircraft at the top of climb (beginning of cruise) point in the mission. You don't want your available rate-of-climb to be 0 in case you need to gain altitude during cruise.
  - `EAS_limit`: the maximum descending EAS in knots.
  - `throttle_setting`: the prescribed throttle setting. This is only used for `GASP` and `solved` missions.
- If a key ends with `_ref` or `_ref0` (except `duration_ref`, `duration_ref0`, `initial_ref` and `initial_ref0`), it is the unit-reference and zero-reference values of the control variable at the nodes. This option is invalid if opt=False. Note that it is a simple usage of  ref and ref0. We refer to [dymos](https://openmdao.github.io/dymos/api/phase_api.html?highlight=ref0#add-state) for details.
- Some keys are for phase time only.
  - `duration_ref` and `duration_ref0` are unit-reference and zero reference for phase time duration.
  - `duration_bounds` are the bounds (lower, upper) for the time duration of the phase.
  - `initial_ref` and `initial_ref0` are the unit-reference and zero references for the initial value of time.
  - `time_initial_ref` and `time_initial_ref0` are the unit-reference and zero-reference for the initial value of time.
  - `initial_bounds`: the lower and upper bounds of initial time. For `GASP`, it is `time_initial_bounds`.
- If a key starts with `final_`, it is the final value of a state variable.
- If a key ends with `_constraint_eq`, it is an equality constraint.

- Keys related to altitude:
  - In `FLOPS` missions, it is `final_altitude`. In GASP missions, it is `final_alt`.
  - Meanwhile, `alt` is a key in acceleration phase parameter for altitude in `GASP` missions and `altitude` is a key in all other phases of all missions.

- Some keys are a boolean flag of True or False:
  - `input_initial`: the flag to indicate whether initial values of of a state (such as: altitude, velocity, mass, etc.) is taken.
  - `add_initial_mass_constraint`: the flag to indicate whether to add initial mass constraint
  - `clean`: the flag to indicate no flaps or gear are included.
  - `connect_initial_mass`: the flag to indicate whether the initial mass is the same as the final mass of previous phase.
  - `fix_initial`: the flag to indicate whether the initial state variables is fixed.
  - `fix_initial_time`: the flag to indicate whether the initial time is fixed.
  - `fix_range`: the flag to indicate whether the range is fixed in this phase.
  - `no_climb`: if True for the descent phase, the aircraft is not allowed to climb during the descent phase.
  - `no_descent`: if True for the climb phase, the aircraft is not allowed to descend during the climb phase.
  - `include_landing`: the flag to indicate whether there is a landing phase.
  - `include_takeoff`: the flag to indicate whether there is a takeoff phase.
  - `optimize_mass`: if True, the gross takeoff mass of the aircraft is a design variable.
  - `target_mach`: the flag to indicate whether to target mach number.
- `initial_guesses`: initial guesses of state variables.
- `COLLOCATION` related keys:
  - `num_segments`: the number of segments in transcription creation in dymos. The minimum value is 1. This is needed if 'AnalysisScheme' is `COLLOCATION`.
  - `order`: the order of polynomials for interpolation in transcription creation in dymos. The minimum value is 3. This is needed if 'AnalysisScheme' is `COLLOCATION`.
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

```{note}
Not all the keys apply to all phases. The users should select the right keys for each phase of interest. The required keys for each phase are defined in [check_phase_info](https://github.com/OpenMDAO/om-Aviary/blob/main/aviary/interface/utils.py) function. Currently, this function does the check only for `FLOPS` and `GASP` missions.
```

Users can add their own keys as needed.
