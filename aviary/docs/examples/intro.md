# Discussing the Aviary Examples

## Current Aviary examples

Aviary provides a range of built-in examples that serve as both regression tests and demonstrations of the tool's capabilities.
These examples showcase various full mission analysis and optimization problems, incorporating different subsystem analyses from FLOPS and GASP.
You can find these examples [here](https://github.com/OpenMDAO/Aviary/tree/main/aviary/validation_cases/benchmark_tests), especially the files that start `test_swap`.
These cases highlight Aviary's ability to replicate GASP and FLOPS capabilities, as well as use both code's methods in a single analysis.

In addition to the examples for core Aviary, we also provide some examples for using external subsystems.
These are contained in the `aviary/examples/external_subsystems` folder.
Currently, the folder contains a few different examples, including a battery external subsystem and an example integration with [OpenAeroStruct](https://github.com/mdolab/OpenAerostruct/).
These examples provide a valuable starting point, especially for users interested in integrating their own subsystems into an Aviary model.

## Future examples

Aviary's collection of examples is expected to undergo significant expansion in the future, particularly as the user interface continues to evolve and more individuals start using the tool.
The current set of examples provides a solid foundation for understanding Aviary's capabilities and serves as a reference for users.
However, there are plans to greatly expand and diversify the examples to cater to a wider range of aircraft analysis and optimization scenarios.
