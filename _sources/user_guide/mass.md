# Mass Subsystem

The mass subsystem in Aviary plays a straightforward but crucial role.
We need to know a reasonable estimate for the mass of an aircraft to evaluate its performance.
The mass subsystem provides this estimate.

```{note}
The mass subsystem in Aviary is similar to the "weights" or "weights and balances" in other tools or aircraft design.
```

## Overview

Aviary's mass subsystem is designed to accommodate different methodologies, including FLOPS-based and GASP-based approaches, catering to diverse design needs and preferences. The subsystem is divided into multiple components, each handling specific aspects of mass calculation.

### FLOPS-Based Components

- **Total Mass Summation:** Orchestrates the calculation of the total mass by summing up contributions from structural, propulsion, systems and equipment, and fuel masses.
- **Structure Mass:** Calculates the mass contributions from the aircraft's structural components like wings, fuselage, landing gear, etc.
- **Propulsion Mass:** Computes the mass related to the aircraft's propulsion system, including engines and associated components.
- **Systems and Equipment Mass:** Determines the mass of systems and equipment on the aircraft, with an alternative calculation option (`AltSystemsEquipMass`) available.
- **Empty Mass:** Represents the total mass of the aircraft without fuel, calculated using either standard or alternative methods.
- **Operating Mass:** Accounts for the mass of the crew, passengers, and service items in addition to the empty mass.
- **Zero Fuel Mass:** The total mass of the aircraft without considering the fuel.
- **Fuel Mass:** Calculates the mass of the fuel required for the mission.

### GASP-Based Components

- **Design Load Group:** Establishes design load parameters that influence other mass calculations.
- **Fixed Mass Group:** Deals with fixed masses, like payload and engine mass, that are essential in determining the wing and fuel mass.
- **Equip and Useful Load Mass:** Calculates the equipment and useful load mass, vital for determining the aircraft's operability.
- **Wing Mass Group:** Computes the mass of the wing, influenced by fixed mass group outputs.
- **Fuel Mass Group:** Determines the fuel mass, taking into account design load and fixed mass group parameters.

## Using the Mass Subsystem

The choice of which code's methods for mass estimate to use is set using the variable `settings:mass_method`. This variable can be specified in the Aviary input file or can be manually set when using the Level 2 or 3 interface.
To effectively use the mass subsystem in Aviary, users need to provide reasonable estimates for mass-related variables in their aircraft .csv file.
Which variables are used depends on which mass estimation subsystem you're using, such as `aircraft:crew_and_payload:mass_per_passenger`, `aircraft:engine:additional_mass_fraction`, etc.

Aviary allows for extensive customization and extensions within the mass subsystem.
Users can develop alternative components and integrate them as subsystems with the existing platform.