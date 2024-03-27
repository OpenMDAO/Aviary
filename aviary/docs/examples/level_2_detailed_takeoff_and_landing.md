# Level 2 Detailed Takeoff and Landing

```{note}
Here we discuss how to optimize the takeoff and landing sequences for aircraft using the Level 2 interface in Aviary.
If you need more precise control over the flight profile and the trajectory definition, please see the Level 3 interface detailed in [FLOPS Based Detailed Takeoff and Landing](../user_guide/FLOPS_based_detailed_takeoff_and_landing.ipynb).
```

This doc page discusses how to use Aviary to perform detailed takeoff and landing simulations for aircraft using the Level 2 interface.
When we say "detailed takeoff and landing," this simply means that we model the aircraft trajectory in more detail than other simplified mission representations.
This means two main things:

- We model the takeoff portion of flight using a series of phases, such as ground roll, rotation, and multiple climb phases. Similarly, we model the landing portion of flight using a series of phases, such as approach, flare, and touchdown.
- Instead of using the height-energy approximation for the aircraft equations of motion, we use the full two-degree-of-freedom (2DOF) equations of motion. This means that there is a notion of angle of attack and aircraft pitch within the flight dynamics equations.

These considerations allow us to model specific parts of the aircraft trajectory in more detail, which is especially useful for certain performance-based disciplinary analyses, such as acoustics and controls.

## How we define the trajectories

Discuss how we use phase_info to define the phases
We generally use polynomial controls of order 1
Any constraints that we need to add we do so in the dictionary
We optimize mach and altitude using the optimize_mach and optimize_altitude flags
You can choose how to enforce the throttle; either solver bounded, with boundary constraints, or path constraints

Initial guesses are important to help the optimizer converge
These guesses are much more important for the 2DOF model than the height-energy model

## Defining the takeoff trajectory

We follow roughly this diagram below

<!-- add figure -->

We have ground roll, rotation, liftoff, and climb phases
At certain points we add constraints and requirements
We don't model the full mission here, just the takeoff portions

## Defining the landing trajectory

We follow roughly this diagram below

<!-- add figure -->

We have approach, flare, and touchdown phases
This is much simpler than takeoff
We don't model the full mission here, just the landing portions