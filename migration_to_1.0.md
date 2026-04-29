# Aviary 1.0 Migration Guide

## 1. General API Changes


## 2. Subsytem API Changes


### `SubsystemBuilderBase`

Has been renamed to `SubsystemBuilder`


### `needs_mission_solver(self, aviary_inputs)`

Arguments changed to `needs_mission_solver(self, aviary_inputs, subsystem_options)`


### `build_pre_mission(self, aviary_inputs, **kwargs)`

Arguments changed to `build_pre_mission(self, aviary_inputs, subsystem_options=None)`


### `get_states(self)`

Arguments changed to `get_states(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `get_controls(self, phase_name=None)`

Arguments changed to `get_controls(self, aviary_inputs=None, user_options=None, subsystem_options=None)`. The "phase_name" argument has been removed in favor of allowing subsystem_options to be directly set in the phase_info.


### `get_parameters(self, aviary_inputs=None, phase_info=None)`

Arguments changed to `get_parameters(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `get_constraints(self)`

Arguments changed to `get_constraints(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `get_linked_variables(self)`

Arguments changed to `get_linked_variables(self, aviary_inputs=None)`


### `get_bus_variables(self)`

This method has been renamed to `get_pre_mission_bus_variables` to distinguish it from `get_post_mission_bus_variables`. The signature is now `get_pre_mission_bus_variables(self, aviary_inputs=None, mission_info=None)`


### `build_mission(self, num_nodes, aviary_inputs, **kwargs)`

Arguments changed to `build_mission(self, num_nodes, aviary_inputs, user_options, subsystem_options)`


### `mission_inputs(self, **kwargs)`

Arguments changed to `mission_inputs(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `mission_outputs(self, **kwargs)`

Arguments changed to `mission_outputs(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `define_order(self)`

This method was never fully implemented and has been removed.

### `get_design_vars(self)`

Arguments changed to `get_design_vars(self, aviary_inputs=None)`


### `get_initial_guesses(self)`

Arguments changed to `get_initial_guesses(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `get_mass_names(self)`

Arguments changed to `get_mass_names(self, aviary_inputs=None)`


### `preprocess_inputs(self, aviary_inputs)`

Arguments changed to `preprocess_inputs(self, aviary_inputs=None)`


### `get_outputs(self)`

This method has been deprecated and replaced with `get_timeseries(self, aviary_inputs=None, user_options=None, subsystem_options=None)`


### `build_post_mission(self, aviary_inputs, **kwargs)`

Arguments changed to `build_post_mission(self, aviary_inputs=None, mission_info=None, subsystem_options=None, phase_mission_bus_lengths=None)`



## 3. Aircraft Data Hieararchy Changes


### 3.1 Renamed Variables

#### `aircraft:controls:total_mass`

Renamed to `aircraft:controls:mass`


#### `aircraft:crew_and_payload:design:num_tourist_class`

Renamed to `aircraft:crew_and_payload:design:num_economy_class`


#### `aircraft:crew_and_payload:non_flight_crew_mass`

Renamed `to aircraft:crew_and_payload:cabin_crew_mass`


#### `aircraft:crew_and_payload:non_flight_crew_mass_scaler`

Renamed `to aircraft:crew_and_payload:cabin_crew_mass_scaler`


#### `aircraft:crew_and_payload:num_tourist_class`

Renamed to `aircraft:crew_and_payload:num_economy_class`


#### `aircraft:crew_and_payload:passenger_mass`

Renamed to `aircraft:crew_and_payload:passenger_mass_total`


#### `aircraft:crew_and_payload:passenger_mass_with_bags`

Renamed to `aircraft:crew_and_payload:mass_per_passenger_with_bags`


#### `aircraft:design:operating_mass`

Renamed to `aircraft:design:operating_mass`


#### `aircraft:design:systems_equip_mass`

Renamed to `aircraft:design:systems_and_equipment_mass`


#### `aircraft:design:systems_equip_mass_base`

Renamed to `aircraft:design:systems_and_equipment_mass_base`


#### `aircraft:design:touchdown_mass`

Renamed to `aircraft:design:touchdown_mass_max`


#### `aircraft:fuselage:avg_diameter`

Renamed to `aircraft:fuselage:ref_diameter`


#### `aircraft:horizontal_tail:vertical_tail_fraction`

Renamed to `aircraft:horizontal_tail:vertical_tail_mount_location`


#### `aircraft:wing:aspect_ratio_ref`

Renamed to `aircraft:wing:aspect_ratio_reference`


#### `aircraft:wing:bwb_aft_body_mass_scaler`

Renamed to `aircraft:wing:bwb_aftbody_mass_scaler`


#### `aircraft:wing:chord_per_semispan`

Renamed to `aircraft:wing:chord_per_semispan_distribution`


#### `aircraft:wing:input_station_dist`

Renamed to `aircraft:wing:input_station_distribution`


#### `aircraft:wing:load_path_sweep_dist`

Renamed to `aircraft:wing:load_path_sweep_distribution`


#### `aircraft:wing:loading`

Renamed to `aircraft:design:wing_loading`


#### `aircraft:wing:surface_ctrl_mass`

Renamed `to aircraft:wing:surface_control_mass`


#### `aircraft:wing:surface_ctrl_mass_scaler`

Renamed `to aircraft:wing:surface_control_mass_scaler`


#### `aircraft:wing:thickness_to_chord_dist`

Renamed to `aircraft:wing:thickness_to_chord_distribution`


#### `mission:design:cruise_altitude`

Renamed to `aircraft:design:cruise_altitude`


#### `mission:design:cruise_range`

Renamed to `aircraft:design:range`


#### `mission:design:gross_mass`

Renamed to `aircraft:design:gross_mass`


#### `mission:design:lift_coefficient`

Renamed to `aircraft:design:lift_coefficient`


#### `mission:design:lift_coefficient_max_flaps_up`

Renamed to `aircraft:design:lift_coefficient_max_flaps_up`


#### `mission:design:mach`

Renamed to `aircraft:design:mach`


#### `mission:design:range`

Renamed to `aircraft:design:range`


#### `mission:design:reserve_fuel`

Renamed to `mission:reserve_fuel`


#### `mission:design:thrust_takeoff_per_eng`

Renamed to `aircraft:design:thrust_takeoff_per_eng`


#### `mission:summary:cruise_mach`

Renamed to `aircraft:design:cruise_mach`


#### `mission:takeoff:fuel_simple`

Renamed to `mission:takeoff:fuel`



### 3.2 Variable Behavior Changes

#### `aircraft:fuselage:num_seats_abreast`

This variable was split into separate variables for each passenger class: `aircraft:fuselage:num_seats_abreast_business`, `aircraft:fuselage:num_seats_abreast_economy`, and `aircraft:fuselage:num_seats_abreast_first`.


#### `aircraft:fuselage:seat_pitch_abreast`

This variable was split into separate variables for each passenger class: `aircraft:fuselage:seat_pitch_abreast_business`, `aircraft:fuselage:seat_pitch_abreast_economy`, and `aircraft:fuselage:seat_pitch_abreast_first`.


#### `mission:summary:fuel_flow_scaler`

This variable was removed. When read from a FLOPS input file, it is incorporated into into `aircraft:engine:subsonic_fuel_flow_scaler` and `aircraft:engine:supersonic_fuel_flow_scaler`.


## 4. Phase_info Format Changes

