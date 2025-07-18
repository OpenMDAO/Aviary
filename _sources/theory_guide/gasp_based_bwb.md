# GASP based BWB model implementation

## BWB geometry and mass

- For design load, variable `Aircraft.Wing.LOADING` is replaced by `Mission.Design.GROSS_MASS / Aircraft.Wing.EXPOSED_WING_AREA`. As a result, `BWBLoadSpeeds` and `BWBLoadFactors` replace `LoadSpeeds` and `LoadFactors`. A new group `BWBDesignLoadGroup` is created to include these two new components.
- Aviary engine geometry uses different empirical equation. In GASP, the sizing relation is based on aircraft gross weight and the number of engines. For BWB, we adopt GASP implementation. We also allow the engines are partially buried into the fuselage. This implementation can be applied to conventional aircraft.

### Equip And Useful Load

  - `EquipAndUsefulLoadMass` is a big components that includes the computations of 19 items. Ideally, each of them should be done in its own component and one group has them all. This is a long time goal. For now, it is separated to two components `EquipMassPartial` and `UsefulLoadMass` and air conditioning and furnishing masses are singled out because they need to be modified for BWB. 
  - A new variable `Aircraft.Electrical.SYSTEM_MASS_PER_PASSENGER` is added which corresponds to `CW(15)` in GASP. Its value is different for conventional aircraft and BWB.
  - Two new classes `BWBACMass` and `BWBFurnishingMass` are added to `equipment_and_useful_load.py`. Unit tests for BWB model are created. Our outputs match with GASP run result.
  - **Note:** GASP Fortran code has new updates that are not included in Aviary. We will update Aviary for furnishing mass but other masses need to be checked.
  - **Note:** `EquipAndUsefulLoadMass` has implementation errors for the computations of `Aircraft.APU.MASS`, `Aircraft.Avionics.MASS`, `Aircraft.AntiIcing.MASS`, `Aircraft.Furnishings.MASS`, and `Aircraft.Design.EMERGENCY_EQUIPMENT_MASS`. As a result, Aviary always uses user provided masses (not empirical formulas). We should use Aviary's feature of `overriding` for thf `overridinose variables. All the outputs of 9 unit tests in `test_mass_summation.py` are updated.

### Wing Mass Model

  - For wing mass, variable `Aircraft.Wing.SPAN` has to deduct cabin width (i.e. `Aircraft.Fuselage.AVG_DIAMETER`). As a result, `WingMassSolve` component is replaced by `BWBWingMassSolve` component. `BWBWingMassGroup` is created to pair `BWBWingMassSolve` and `WingMassTotal`.
  - In `geometry/gasp_based/wing.py`, `Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX` is moved out of `WingParameters` class. In stead, a class `WingVolume` is created to compute `Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX`. For BWB, another class `BWBWingVolume` is created for the same purpose. The algorithm is quite different for BWB.
  - In `geometry/gasp_based/wing.py`, `WingFold` class is split to two: `WingFoldArea` and `WingFoldVolume`. The first computes `Aircraft.Wing.FOLDING_AREA` and the second computes `Aircraft.Fuel.WING_VOLUME_GEOMETRIC_MAX`. For BWB, another class `BWBWingFoldVolume` is created to do the same job. Note that for BWB, `BWBWingFoldVolume` uses the result in `BWBWingVolume`.
  - A `BWBWingGroup` is created to put all these pieces together. 
  - Unit tests are added in  `geometry/gasp_based/test/test_wing.py` to make sure that the Aviary result is the same as GASP model.

### Fuel Model

  - `FuelMassGroup` groups all fuel related components. In case of BWB, `BWBFuselageMass` is in place of `FuselageMass`. This group has a nonlinear solver. In order for it to converge, one must provide good initial guesses for the inputs. Otherwise, it may claim that convergence is reached but gives rise to a strange solution.
  - The computation in `BodyTankCalculations` of `fuel.py` can not be matched in GASP Fortran code. It is not very clear if it is correct. It is possible that `extra_fuel_volume` becomes negative. I added code to make sure that it is always positive.
  - For fuselage mass, the empirical weight equation is quite different. It is computed in `FuselageAndStructMass` component. This component has two parts: fuselage mass and structural mass. In order to reuse the code for structural mass, this component is split into two components: `FuselageMass` and `StructMass`. For BWB, `FuselageMass` is replaced by `BWBFuselageMass`.
  - **Note:** The historic name of `Mission.Design.FUEL_MASS_REQUIRED` is `INGASP.WFAREQ`, but `WFAREQ` includes fuel margin in GASP while `Mission.Design.FUEL_MASS_REQUIRED` doesn't. The historic name of `Mission.Summary.TOTAL_FUEL_MASS` is `INGASP.WFA`, but does not include fuel margin in GASP while `Mission.Design.FUEL_MASS_REQUIRED` does.
  - **Note:** GASP Fortran code has items that are not implemented in Aviary (e.g. tail boom support, tip tank weight, fuselage acoustic treatment, pylon, acoustic treatment).

- Comparison to GASP model

| Aviary | &nbsp; &nbsp; | GASP | &nbsp; &nbsp; | Observation |
| ------- | ------- | ------- | -------- | ------------- |
| Aircraft.Design.FIXED_USEFUL_LOAD | 5972 | WFUL | 5775 | different unit weight of pilots and attendents |
| Aircraft.Wing.HIGH_LIFT_MASS | 972 | WHLDEV | 974 | In GASP, wing loading is a variable, but in Aviary, it is a constant |
| Aircraft.Fuel.FUEL_SYSTEM_MASS | 760 | WFSS | 1281 | the mass in GASP is computed after engine sizing. |
| Aircraft.Design.STRUCTURE_MASS | 44471 | WST | 45623 | the mass in GASP is computed after engine sizing. |
| Mission.Design.FUEL_MASS | 19744 | WFADES | 33268 | BodyTank algorithm is different and the mass in GASP is computed after engine sizing. |
| Aircraft.Propulsion.MASS | 8072 | WP | 8592 | the mass in GASP is computed after engine sizing. |
| Aircraft.Fuel.TOTAL_CAPACITY | 19744 | WFAMAX | 33268 | BodyTank algorithm is different and the mass in GASP is computed after engine sizing. |
| | | | |

Because most of the variables match pretty well. We show those with significant differences. As we see, the fuel related masses are quite different from GASP. This is mainly due to the fact that GASP computes the fuel masses after engine sizing.

## BWB aerodynamics

This feature implements GASP aerodynamics subsystems for BWB aircraft. Five new components are added:

- `BWBBodyLiftCurveSlope`
- `BWBFormFactorAndSIWB`
- `BWBAeroSetup`
- `BWBLiftCoeff`
- `BWBLiftCoeffClean`

Two group components `CruiseAero` and `LowSpeedAero` are configured for BWB as an option (default to conventional aircraft).

In GASP, friction due to nacelle is removed from the computation of `SA5`. Instead, it is computed separately and is added in the drag computation.

`alpha_stall` and `CL_max` are computed based on wing only. We expect that they will be updated in the future.

Table based aerodynamics is still available to BWB. A new table based BWB model will be provided in the future.

### Comparison of `BWBAeroSetup` with GASP

| Variables | GASP | Variables | Aviary |
| ---------- | ------ | ------- | ------- |
| CLAW | 4.63868 | lift_curve_slope | 4.63868 |
| BARL | -0.14081 | lift_ratio | -0.14081 |
| CFIN | 0.002836 | cf | 0.002836 |
| SA1 | 0.81401 | SA1 | 0.80832 |
| SA2 | -0.15743 | SA2 | -0.13651 |
| SA3 | 0.033989 | SA3 | 0.033989 |
| SA4 | 0.10197 | SA4 | 0.10197 |
| SA5 | 0.004464 | SA5 | 0.009628 |
| SA6 | 2.23877 | SA6 | 2.09277 |
| SA7 | 0.034136 | SA7 | 0.040498 |

The differences are due to several reasons:

- GASP has different coefficients of friction for different part of an aircraft. For this purpose, several new parameters (aero calibration factors) are added. Aviary has one single coefficient `cf`.
- GASP has several factors that are included in the computation of friction (e.g. winglet, tip tank, excrescence) but not in Aviary.
- GASP excludes frictions from nacelle in `SA5`. Nacelle friction is done in engine computation and is added in drag computation. But in Aviary, nacelle friction is included in `SA5` and not in drag computation.

### Comparison of `CruiseAero` with GASP

| Variables | GASP | Variables | Aviary |
| ---------- | ------ | ------- | ------- |
| CL | 0.41069 | CLTOT | 0.41067 |
| CD | 0.014738 | CD | 0.022509 |
| CL/CD | 27.86518 | L/D | 18.24451 |

As we see, `CL` matches closely but `CD` doesn't. This is because the differences in  `BWBAeroSetup`.

### Comparison of `LowSpeedAero` with GASP

| Variables (Takeoff) | CL (GASP/Aviary) | CD (GASP/Aviary)| CL/CD (GASP/Aviary) |
| --------------- | -----------------  | ------------------- | ----------------------- |
| α = -2.0 | 0.07507 / 0.05787 | 0.01853 / 0.02565 | 4.05136 / 3.307513 |
| α = 0.0 | 0.23964 / 0.21906 | 0.01866 / 0.02592 | 12.84433 / 9,49165 |
| α = 2.0 | 0.40422 / 0.407231 | 0.02070 / 0.02844 | 20.74018 / 16.22583 |

| Variables (Landing) | CL (GASP/Aviary) | CD (GASP/Aviary)| CL/CD (GASP/Aviary) |
| --------------- | -----------------  | ------------------- | ----------------------- |
| α = -2.0 | 0.18551 / 0.19824 | 0.02299 / 0.02962 | 8.06918 / 6.69194 |
| α = 0.0 | 0.35009 / 0.35944 | 0.02292 / 0.02970 | 15.27225 / 12.10145 |
| α = 2.0 | 0.51467 / 0.52062 | 0.02482 / 0.03209 | 20.74018 / 16.22583 |

As we see, `CL` matches closely but `CD` doesn't. This is because the differences in  `BWBAeroSetup`.

## Missing features in Aviary

In addition to the missing fetures, there are other features in GASP that are not implemented in Aviary.

- GASP computes maximum CL for cruise, take-off, and landing phases but not in Aviary.
- GASP computes lift curve slope (i.e. the derivative of the Lift Coeff w.r.t. Alpha), named `CLATOT`. 
- GASP computes stall alpha from the wings. For BWB, this is not sufficient. Both GASP and Aviary should enhance their models.
- GASP computes excrescence drag.
- Drag coefficients SA3 and SA4 are computed in Aviary but are not used.
- Aviary does not have tail boom support.
- Aviary does not have winglet geometry.
- In GASP, a pilot weight is 170 lb and in Aviary it is 198 lb. In GASP, each attendant weights 130 lb and in Aviary it is 177 lb.
- GASP has fuselage acoustic treatment.
- GASP conputes tip tank weight.
- GASP allows canard configurations.
