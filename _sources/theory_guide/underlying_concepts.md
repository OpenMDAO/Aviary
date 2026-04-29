# Underlying Concepts

```{note}
Much of this doc page has not been developed yet.
```

## Aircraft Design Theory

Aviary is a tool for aircraft design that is largely based on prior NASA-developed tools such as FLOPS and GASP.
Most of the underlying models and methods draw from textbook-based conceptual and preliminary design concepts.

Aviary allows aircraft design through many different design variables, including wing, fuselage, and tail sizing, propulsion system sizing, and weight analysis.
We will not go into great detail here on the theory behind these models as later doc pages will cover these topics in more detail.

## Mission Optimization Theory

Mission optimization in aircraft design plays a pivotal role in enhancing the overall performance and capabilities of aircraft.
It involves the intricate balancing of numerous variables and constraints to achieve optimal flight trajectories and mission outcomes.

Notably, mission optimization is not just about the literal path an aircraft takes, but encompasses a broader range of state histories.
These include short-term effects like battery state of charge, component temperatures, and propulsion system states.
Therefore, mission optimization necessitates a comprehensive approach that considers these dynamic aspects.

In this context, we introduce several key terms: states, controls, design parameters, and constraints.
States refer to values tracked over time, like fuel levels or component temperatures.
Controls are values varied over time to influence the system, such as throttle settings or control surface deflections.
Design parameters, or static controls, are fixed variables like wingspan or engine size.
Constraints are system values we aim to limit at certain times, like maximum Mach speeds, altitude constraints, or battery charge constraints.

The complexity of mission optimization increases with the flexibility required to optimize these variables.
For a robust optimization, it's crucial to consider an aircraft's performance across its entire mission trajectory.
This approach allows for the tracking of path-dependent states, thereby offering a more accurate performance evaluation.
However, this typically requires numerous model evaluations, underscoring the complexity of the task.

Previous studies in trajectory analysis and optimization, particularly those incorporating physics-based analyses, are crucial in this field.
NASA Glenn researcher Rob Falck has published several papers on mission optimization, especially in the context of systems design and multidisciplinary design optimization.
His [2019 paper on Optimal Control within the Context of Multidisciplinary Design, Analysis, and Optimization](https://ntrs.nasa.gov/api/citations/20190002793/downloads/20190002793.pdf) is a great and digestible introduction to a lot of the mission optimization concepts contained within Aviary.
For a more specific example of mission optimization for a specific aircraft platform, his paper on [Trajectory Optimization of Electric Aircraft Subject to Subsystem Thermal Constraints](https://ntrs.nasa.gov/api/citations/20170009148/downloads/20170009148.pdf) is a great read.
Lastly, although the author of this doc page is slightly biased, [John Jasa's dissertation](https://deepblue.lib.umich.edu/bitstream/handle/2027.42/155269/johnjasa_1.pdf?sequence=1&isAllowed=y) also has a great overview of mission optimization concepts, particularly for path-dependent optimization problems.
Sections 1.2, 2.1, and 2.2 are especially relevant if you want to dig more into the history behind mission optimization and other related topics.

Although these studies offer a foundation, it's important to acknowledge the real-world constraints and regulations that might affect aircraft operation.
These practical considerations sometimes differ from theoretical models and must be taken into account for realistic mission planning.
Aviary has some built-in constraints to account for these factors, though many FAA or FAR regulations are not necessarily considered.

In summary, mission optimization in aircraft design is a multifaceted challenge that demands a thorough understanding of the system-to-system interactions on board an aircraft.
Aviary aims to provide a robust platform for constructing and solving coupled aircraft-mission design problems.

## Design in Aviary

This section is under development.

## Design vs. Off Design

This section is under development.
