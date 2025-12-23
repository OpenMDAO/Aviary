from aviary.mission.flight_phase_builder import FlightPhaseBase, register
from aviary.mission.height_energy.ode.energy_ODE import EnergyODE
from aviary.mission.initial_guess_builders import InitialGuessIntegrationVariable, InitialGuessState


# TODO: support/handle the following in the base class
# - phase.set_time_options()
#     - currently handled in level 3 interface implementation
# - self.external_subsystems
# - self.meta_data, with cls.default_meta_data customization point
@register
class EnergyPhase(FlightPhaseBase):
    default_ode_class = EnergyODE


EnergyPhase._add_initial_guess_meta_data(
    InitialGuessIntegrationVariable(),
    desc='initial guess for initial time and duration specified as a tuple',
)

EnergyPhase._add_initial_guess_meta_data(
    InitialGuessState('distance'), desc='initial guess for horizontal distance traveled'
)
