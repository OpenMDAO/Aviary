import openmdao.api as om

from aviary.mission.flops_based.phases.simplified_landing import LandingGroup
from aviary.variable_info.variables import Aircraft, Mission


class Landing:
    """
    Define user constraints for a climb phase.

    Parameters
    ----------
    ref_wing_area : float (0.0)
        reference are of wing (ft^2)
    Cl_max_ldg : float (None)
        maximum coefficient of lift in landing configuration (None)

    Returns
    -------
    Group
        a Group object in OpenMDAO
    """

    def __init__(
        self,
        ref_wing_area=None,
        Cl_max_ldg=None,
    ):
        self.ref_wing_area = ref_wing_area  # ft**2
        self.Cl_max_ldg = Cl_max_ldg  # no units
        # note: need to clean this up so that some of these variables come from
        # connections.

    __slots__ = (
        "ref_wing_area",
        "Cl_max_ldg",
    )

    def build_phase(self, use_detailed=False):
        """
        Construct and return a new phase for landing analysis.
        Parameters
        ----------
        use_detailed : bool (False)
            tells whether to use simplified or detailed landing. Currently detailed is
            disabled.
        Returns
        -------
        Group
            a group in OpenMDAO
        """

        if use_detailed:
            raise om.AnalysisError(
                "Must set landing method to `use_detailed=False`, detailed landing is"
                " not currently enabled."
            )

        ##############
        # Add Inputs #
        ##############

        landing = LandingGroup()
        landing.set_input_defaults(
            Aircraft.Wing.AREA, val=self.ref_wing_area, units="ft**2"
        )
        landing.set_input_defaults(
            Mission.Landing.LIFT_COEFFICIENT_MAX, val=self.Cl_max_ldg, units='unitless')

        return landing
