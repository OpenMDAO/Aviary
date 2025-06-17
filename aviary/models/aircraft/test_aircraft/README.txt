The aircraft in this folder are variations of the existing example models intended for testing Aviary's code.
The models themselves are valid Aviary aircraft that produce physically possible results, but may not be "realistic" due to arbitrary modifications.

The suite of test aircraft include:
- `aircraft_for_bench_FwFm`: conventional single-aisle transport using all FLOPS-based analysis
- `aircraft_for_bench_FwGm`: conventional single-aisle transport using FLOPS mass, GASP aero, both geometry methods (FLOPS priority), and the 2DOF mission method
- `aircraft_for_bench_GwGm`: conventional single-aisle transport using all GASP-based analysis
- `aircraft_for_bench_GwFm`: conventional single-aisle transport using GASP mass, geometry, and aero, and the energy mission method
- `aircraft_for_bench_FwFm_with_electric`: the FwFm benchmark model using the electrified 28k turbofan engine (no batteries/external subsystems)
- `aircraft_for_bench_solve2dof`: the GwGm benchmark using the "solved 2dof" equations of motion

Also included in this folder:
`GwFm_phase_info`: the custom phase info needed for the GwFm benchmark
`turbofan_28k_with_electric`: a version of the "turbofan_28k" engine deck with arbitrary electric power consumption. Used for the "aircraft_for_bench_FwFm_with_electric" model.

