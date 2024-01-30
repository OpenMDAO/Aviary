# What Aviary Does

"Why make a new aircraft design tool?" is probably a valid question to ask.
We'll keep it short and sweet.

Aviary is an *open-source* tool that allows users to:

- design aircraft and optimize trajectories simultaneously
- add their own custom subsystems
- use gradient-based optimization effectively

While there are other tools that can do some of these things, we've designed Aviary to do *all* of these things in an approachable way.
Aviary is being used in multiple NASA projects across multiple centers and we're excited to share it with the world.

Let's discuss what Aviary does in more detail.

```{warning}
Aviary is under active development!
If you're using it, know that we are working to update Aviary to make it more user-friendly and capable.
If you have suggestions or comments please let the Aviary team know by [submitting an issue on GitHub](https://github.com/OpenMDAO/Aviary/issues/new/choose).
```

## Core functionalities

### Core subsystems

The core functionalities of Aviary revolve around preliminary and conceptual-level aircraft design.
Aviary includes a suite of core subsystems that are needed for aircraft design.
These include:

- aerodynamics
- propulsion
- mass
- geometry
- mission analysis

For each of these subsystems, Aviary provides a few different models that can be used.
In broad strokes, these include empirical models based on historical data, tabular data from other models or experiments, and some physics-based models.

The [Theory Guide](../theory_guide/intro) provides much more information on these subsystems.

### Pre-mission, Mission, and Post-mission

Aviary problems consist of three sections: pre-mission, mission, and post-mission.
The pre-mission portion is where the aircraft is sized and any necessary pre-computed values needed for the mission are generated.
The mission portion models the aircraft flight through a trajectory, evaluating the performance along the mission at "nodes" (points in time).
The post-mission portion is where the performance metrics are calculated and any postprocessing occurs.

For more information on these sections, please see the [pre-mission and mission doc page](../user_guide/pre_mission_and_mission).

### Custom Subsystems

One of the main goals of Aviary is to allow users to add their own custom subsystems.
The point is to make this as easy as possible while providing enough flexibility for complicated subsystems to be added to Aviary problems.
We have already made quite a few subsystems for projects internal to NASA.
These subsystems include a hybrid-electric propulsion system using [pyCycle](https://github.com/OpenMDAO/pyCycle), an aerostructural wing model using both [TACS](https://github.com/smdogroup/tacs) and [OpenAeroStruct](https://github.com/mdolab/OpenAeroStruct/), economic models to estimate the cost of the aircraft, and many more.

Please see the [external subsystems doc pages](../user_guide/subsystems) for more information on how to add your own subsystems.

## Types of aircraft and missions it can design

Aviary is designed to be flexible enough to design a variety of aircraft.
Although many of the core subsystems were written with transport aircraft in mind, these subsystems can be replaced or augmented to support other types of aircraft.
We have used Aviary to design single- and double-aisle commercial aircraft, regional jets, advanced concepts like truss-braced wings with hybrid-electric propulsion, and more.

Aviary is also able to optimize a variety of mission types and definitions.
Users are able to freely define the mission definition, including the number of phases, which quantities are controlled by the optimizer, and more.

The most simplistic realistic missions include takeoff, climb, cruise, descent, and landing.
Aviary can also optimize more complicated missions, including those with multiple climb phases, cruise-climbs, multiple descents, etc.
If you're only interested in a single phase, Aviary can optimize that as well; you don't always have to evaluate a full mission.

## Benefits and limitations

There are a bevy of benefits and a few limitations to consider when using Aviary.
We've touched on some of these already, but let's discuss them more.

### Benefits

- Open-source: Aviary is open-source and available on [GitHub](https://github.com/OpenMDAO/Aviary)
- Flexible: Aviary is intended to be flexible enough to design a variety of aircraft and missions
- Customizable: Aviary allows users to add their own subsystems to the problem
- Optimization: Aviary is designed to be used effectively with gradient-based optimization
- Python: Aviary is written in Python and uses [OpenMDAO](https://openmdao.org/) as the underlying framework

### Limitations

- Levels of Fidelity: Aviary is designed for preliminary and conceptual-level aircraft design, not detailed design
- Physics-based Models: Aviary does not have many physics-based models
- Computational Speed: Aviary's focus is not computational speed, as you might've guessed by it being written in Python. Instead, we hope it is more approachable and flexible than other tools.

For other information on some of the limitations and future work related to those limitations, please see the [Planned Future Features doc page](../misc_resources/planned_future_features).
