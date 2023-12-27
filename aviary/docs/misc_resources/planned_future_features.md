# Planned Future Features

Aviary is under active development and new features are being added regularly.
The following is a non-exhaustive list of planned features that are not yet implemented. (The Aviary team reserves the right to remove features from this list if the need arises. This list is provided for informational purposes only and is not a commitment to perform work.)

- The ability to run off-design missions to develop payload-range diagrams
- The full capabilities of FLOPS and GASP to model medium and large sized commercial aircraft, with a few exceptions that have been determined unnecessary
- The tested ability to have different types of engines on the same aircraft
- The ability to accept propeller maps for modeling propeller-driven aircraft
- A converter to convert a table of mach/altitude combinations into a phase_info file. This capability exists in the simple mission GUI, but it will be tweaked to allow for the direct input of tabular data as an alternative to the GUI
- Natively supported builders for certain high-interest external subsystem tools. Some potential tools to support are: NPSS, pyCycle, OpenVSP, VSPaero, OpenAeroStruct, etc. The Aviary team develops these builders as the need arises, and the development or lack of it for a certain tool does not indicate endorsement of the tool.
- Improved cleanliness of the code
- Improved ease-of-use for the user interface
- Improved Fortran-to-Aviary converter which requires no human intervention or checking
- Support for relevant FAA regulations governing aircraft design and operation
- Capability to fly reserve missions using the same mission analysis techniques as the main mission (right now reserve estimates are fixed values or fixed percentages of mission fuel)
- Improved model re-run capability
- Full test suite that tests the code format, including testing for docstrings on all functions and classes
- Fully tested code blocks in the documentation