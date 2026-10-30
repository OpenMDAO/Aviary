import openmdao.api as om

from aviary.mission.energy_state.phases.simplified_landing import LandingGroup


class Landing:
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
                'Must set landing method to `use_detailed=False`, detailed landing is'
                ' not currently enabled.'
            )
        else:
            # Simple landing group
            landing = LandingGroup()

        return landing
