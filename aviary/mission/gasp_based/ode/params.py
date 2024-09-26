import openmdao.api as om

from aviary.variable_info.variables import Aircraft, Mission


class ParamPort(om.ExplicitComponent):
    """
    Component that adds variables needed by mission systems to the OpenMDAO problem
    This is to be replaced with curated lists in the areo and propulsion builders using
    the get_parameters() method.
    """

    param_data = {
        Aircraft.Wing.INCIDENCE: dict(units="deg", val=0),
        Aircraft.Wing.FLAP_DEFLECTION_TAKEOFF: dict(units="deg", val=10),
        Aircraft.Wing.FLAP_DEFLECTION_LANDING: dict(units="deg", val=40),
    }

    def setup(self):
        # override
        for name, data in self.param_data.items():
            self.add_input(name, **data)

    @staticmethod
    def set_default_vals(sys):
        """Set input defaults on a group"""
        for name, data in ParamPort.param_data.items():
            sys.set_input_defaults(
                name, val=data.get("val", 1), units=data.get("units", None)
            )

    @staticmethod
    def add_params(param_data):
        """Overlay additional parameters for specialized dynamic systems.

        Parameters
        ----------
        param_data : dict
            Dictionary with parameter names as keys and sub-dictionaries for values,
            containing optionally ``val`` and ``units`` keys to set default value and
            units for the parameter.
        """
        ParamPort.param_data.update(param_data)

    @staticmethod
    def promote_params(sys, trajs=None, phases=None):
        """Add parameters to trajectories and/or phases and set defaults.

        Parameters are also promoted at the ``sys`` level.

        If trajectories are provided, the phases for each trajectory must also be
        supplied. If only phases are provided, no trajectory-level parameters are added.

        Parameters
        ----------
        sys : System
            System of which the trajectories or phases are subsystems.
        trajs : iterable, optional
            Trajectories to which to add parameters. If ``None`` (default), only phase
            parameters are added.
        phases : iterable, optional
            Phases to which to add parameters.
        """
        proms = [(f"parameters:{param}", param) for param in ParamPort.param_data]

        if trajs:
            for trajname, phasenames in zip(trajs, phases):
                traj = getattr(sys, trajname)
                for param, data in ParamPort.param_data.items():
                    traj.add_parameter(
                        param,
                        units=data.get("units", None),
                        val=data.get("val", 1.0),
                        opt=False,
                        static_target=True,
                        targets={phasename: [param] for phasename in phasenames},
                    )
                sys.promotes(trajname, inputs=proms)
        else:
            for phasename in phases:
                phase = getattr(sys, phasename)
                for param, data in ParamPort.param_data.items():
                    phase.add_parameter(
                        param,
                        units=data.get("units", None),
                        val=data.get("val", 1.0),
                        opt=False,
                        static_target=True,
                    )
                sys.promotes(phasename, inputs=proms)


params_for_unit_tests = {
    Aircraft.Wing.AREA: dict(units="ft**2", val=1370.3),
    Aircraft.Wing.HEIGHT: dict(units="ft", val=8),
    Aircraft.Wing.SPAN: dict(units="ft", val=117.8),
    Mission.Design.GROSS_MASS: dict(units="lbm", val=175400),
    Mission.Summary.GROSS_MASS: dict(units="lbm", val=175400),
    Mission.Summary.FUEL_FLOW_SCALER: dict(units="unitless", val=1.0),
    Mission.Takeoff.AIRPORT_ALTITUDE: dict(units="ft", val=0),
    Mission.Landing.AIRPORT_ALTITUDE: dict(units="ft", val=0),
    Aircraft.Wing.AVERAGE_CHORD: dict(units="ft", val=12.615),
    Aircraft.Fuselage.AVG_DIAMETER: dict(units="inch", val=12 * 13.100),
    Aircraft.HorizontalTail.AVERAGE_CHORD: dict(units="ft", val=9.577),
    Aircraft.HorizontalTail.AREA: dict(units="ft**2", val=375.880),
    Aircraft.HorizontalTail.SPAN: dict(units="ft", val=42.254),
    Aircraft.VerticalTail.AVERAGE_CHORD: dict(units="ft", val=16.832),
    Aircraft.VerticalTail.AREA: dict(units="ft**2", val=469.318),
    Aircraft.VerticalTail.SPAN: dict(units="ft", val=27.996),
    Aircraft.Fuselage.LENGTH: dict(units="ft", val=129.4),
    Aircraft.Nacelle.AVG_LENGTH: dict(units="ft", val=14.5),
    Aircraft.Fuselage.WETTED_AREA: dict(units="ft**2", val=4000),
    Aircraft.Nacelle.SURFACE_AREA: dict(units="ft**2", val=659.23 / 2),
    Aircraft.Wing.THICKNESS_TO_CHORD_UNWEIGHTED: dict(units="unitless", val=0.1397),
    Aircraft.Strut.CHORD: dict(
        units="ft", val=0
    ),  # only available if Aviary_option Aircraft.Wing.HAS_STRUT
    Aircraft.Wing.ASPECT_RATIO: dict(units="unitless", val=10.13),
    Aircraft.Wing.TAPER_RATIO: dict(units="unitless", val=0.33),
    Aircraft.Wing.THICKNESS_TO_CHORD_ROOT: dict(units="unitless", val=0.15),
    Aircraft.Wing.THICKNESS_TO_CHORD_TIP: dict(units="unitless", val=0.12),
    Aircraft.HorizontalTail.VERTICAL_TAIL_FRACTION: dict(units="unitless", val=0),
    Aircraft.Wing.SWEEP: dict(units="deg", val=25),
    Aircraft.HorizontalTail.SWEEP: dict(units="deg", val=25),
    Aircraft.HorizontalTail.MOMENT_RATIO: dict(units="unitless", val=0.2307),
    Aircraft.Wing.MOUNTING_TYPE: dict(units="unitless", val=0),
    Aircraft.Design.STATIC_MARGIN: dict(units="unitless", val=0.03),
    Aircraft.Design.CG_DELTA: dict(units="unitless", val=0.25),
    Aircraft.Wing.FORM_FACTOR: dict(units="unitless", val=1.25),
    Aircraft.Fuselage.FORM_FACTOR: dict(units="unitless", val=1.25),
    Aircraft.Nacelle.FORM_FACTOR: dict(units="unitless", val=1.5),
    Aircraft.VerticalTail.FORM_FACTOR: dict(units="unitless", val=1.25),
    Aircraft.HorizontalTail.FORM_FACTOR: dict(units="unitless", val=1.25),
    Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR: dict(units="unitless", val=1.1),
    Aircraft.Strut.FUSELAGE_INTERFERENCE_FACTOR: dict(units="unitless", val=0),
    Aircraft.Design.DRAG_COEFFICIENT_INCREMENT: dict(units="unitless", val=0.00175),
    Aircraft.Fuselage.FLAT_PLATE_AREA_INCREMENT: dict(units="ft**2", val=0.25),
    Aircraft.Wing.CENTER_DISTANCE: dict(units="unitless", val=0.463),
    Aircraft.Wing.MIN_PRESSURE_LOCATION: dict(units="unitless", val=0.3),
    Aircraft.Wing.MAX_THICKNESS_LOCATION: dict(units="unitless", val=0.4),
    Aircraft.Strut.AREA_RATIO: dict(units="unitless", val=0),
    Aircraft.Wing.ZERO_LIFT_ANGLE: dict(units="deg", val=-1.2),
    Aircraft.Design.SUPERCRITICAL_DIVERGENCE_SHIFT: dict(units="unitless", val=0.033),
    Aircraft.Wing.FLAP_CHORD_RATIO: dict(units="unitless", val=0.3),
    Mission.Design.LIFT_COEFFICIENT_MAX_FLAPS_UP: dict(units="unitless", val=1.2596),
    Mission.Takeoff.LIFT_COEFFICIENT_MAX: dict(units="unitless", val=2.1886),
    Mission.Landing.LIFT_COEFFICIENT_MAX: dict(units="unitless", val=2.8155),
    Mission.Takeoff.LIFT_COEFFICIENT_FLAP_INCREMENT: dict(
        units="unitless", val=0.4182
    ),
    Mission.Landing.LIFT_COEFFICIENT_FLAP_INCREMENT: dict(
        units="unitless", val=1.0293
    ),
    Mission.Takeoff.DRAG_COEFFICIENT_FLAP_INCREMENT: dict(
        units="unitless", val=0.0085
    ),
    Mission.Landing.DRAG_COEFFICIENT_FLAP_INCREMENT: dict(
        units="unitless", val=0.0406
    ),
}


def set_params_for_unit_tests(prob):
    """
    Helper function to set parameters for several ode tests with the 2DOF method.

    This is needed because the Paramport used to contain default values for some
    variables.

    Parameters
    ----------
    prob : Problem
        OpenMDAO problem that has been setup.

    Returns
    -------
    Problem
    """
    for key, val in params_for_unit_tests.items():
        try:
            prob.set_val(key, val['val'], units=val['units'])
        except:
            pass

    return prob
