import openmdao.api as om

from aviary.mission.flops_based.phases.simplified_takeoff import TakeoffGroup
from aviary.variable_info.variables import Dynamic


class Takeoff:
    """
    Define user constraints for a climb phase.

    Parameters
    ----------
    airport_altitude : float (None)
        altitude of airport (ft)
        ratio of lift to drag at takeoff (None)
    num_engines int (None)
        number of engines (None)

    Returns
    -------
    Group
        a Group object in OpenMDAO
    """

    def __init__(
        self,
        airport_altitude=None,
        ramp_mass=None,
        num_engines=None,
    ):
        self.airport_altitude = airport_altitude  # ft
        self.num_engines = num_engines
        # note: need to clean this up so that some of these variables come from
        # connections. The only variables that should stay are: airport_altitude

    __slots__ = (
        "airport_altitude",
        "num_engines",
    )

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
                "Must set takeoff method to `use_detailed=False`, detailed takeoff is"
                " not currently enabled."
            )

        ##############
        # Add Inputs #
        ##############

        takeoff = TakeoffGroup(num_engines=self.num_engines)
        takeoff.set_input_defaults(
            Dynamic.Mission.ALTITUDE,
            val=self.airport_altitude,
            units="ft")

        return takeoff
