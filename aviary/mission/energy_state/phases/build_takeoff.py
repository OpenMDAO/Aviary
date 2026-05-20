import openmdao.api as om

from aviary.mission.energy_state.phases.simplified_takeoff import TakeoffGroup
from aviary.variable_info.variables import Dynamic


class Takeoff:
    """
    Define user constraints for a takeoff phase.

    Parameters
    ----------
    airport_altitude : float (None)
        altitude of airport (ft)
        ratio of lift to drag at takeoff (None)

    Returns
    -------
    Group
        a Group object in OpenMDAO
    """

    def __init__(
        self,
        airport_altitude=None,
    ):
        self.airport_altitude = airport_altitude  # ft

    __slots__ = 'airport_altitude'

    def build_phase(self, use_detailed=False):
        """
        Construct and return a new phase for takeoff analysis.

        Parameters
        ----------
        use_detailed : bool(False)
            tells whether to use simplified or detailed takeoff. Currently detailed is
            disabled.

        Returns
        -------
        Group
            a group in OpenMDAO
        """
        if use_detailed:  # TODO
            raise om.AnalysisError(
                'Must set takeoff method to `use_detailed=False`, detailed takeoff is'
                ' not currently enabled.'
            )

        ##############
        # Add Inputs #
        ##############

        takeoff = TakeoffGroup()
        takeoff.set_input_defaults(Dynamic.Mission.ALTITUDE, val=self.airport_altitude, units='ft')

        return takeoff
