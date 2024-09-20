import numpy as np

import openmdao.api as om

from aviary.mission.ode.specific_energy_rate import SpecificEnergyRate
from aviary.mission.ode.altitude_rate import AltitudeRate
from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import AnalysisScheme, AlphaModes, SpeedType
from aviary.variable_info.variables import Aircraft, Dynamic


class BaseODE(om.Group):
    """
    The base class for all GASP based ODE components.
    """

    def initialize(self):
        self.options.declare("num_nodes", default=1, types=int)
        self.options.declare(
            "analysis_scheme",
            default=AnalysisScheme.COLLOCATION,
            types=AnalysisScheme,
            desc="The analysis method that will be used to close the trajectory; for example collocation or time integration",
        )

        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

        self.options.declare(
            'core_subsystems',
            desc='list of core subsystems'
        )

        self.options.declare(
            'subsystem_options', types=dict, default={},
            desc='dictionary of parameters to be passed to the subsystem builders'
        )

    def AddAlphaControl(
        self,
        alpha_group=None,
        alpha_mode: AlphaModes = AlphaModes.DEFAULT,
        num_nodes=1,
        target_load_factor=1.1,
        target_tas_rate=0,
        # target_alt_rate=0,
        # target_flight_path_angle=0,
        atol=1e-7,
        rtol=1e-7,
        add_default_solver=True,
        print_level=0,
    ):
        '''
        This is used when angle of attack in an ODE needs to be controlled directly.
        '''

        if not alpha_group:
            alpha_group = self
        nn = num_nodes

        if alpha_mode is AlphaModes.ROTATION:
            alpha_comp = om.ExecComp(
                'alpha=rotation_rate*(t_curr-start_rotation)+alpha_init',
                alpha=dict(val=0., units='deg'),
                rotation_rate=dict(val=10.0/3.0, units='deg/s'),
                t_curr=dict(val=0., units='s'),
                start_rotation=dict(val=0., units='s'),
                alpha_init=dict(val=0., units='deg'),
            )
            alpha_comp_inputs = ["rotation_rate", "t_curr", "start_rotation",
                                 ("alpha_init", Aircraft.Wing.INCIDENCE)]

        elif alpha_mode is AlphaModes.LOAD_FACTOR:
            alpha_comp = om.BalanceComp(
                name="alpha",
                val=np.full(nn, 10),  # initial guess
                units="deg",
                eq_units="unitless",
                lhs_name="load_factor",
                rhs_val=target_load_factor,
                upper=25.,
                lower=-2.,
            )
            alpha_comp_inputs = ["load_factor"]

        elif alpha_mode is AlphaModes.FUSELAGE_PITCH:
            alpha_comp = om.ExecComp(
                'alpha=max_fus_angle-gamma+i_wing',
                alpha=dict(val=0., units='deg'),
                max_fus_angle=dict(val=0., units='deg'),
                gamma=dict(val=0., units='deg'),
                i_wing=dict(val=0., units='deg'),
            )
            alpha_comp_inputs = [("max_fus_angle", Aircraft.Design.MAX_FUSELAGE_PITCH_ANGLE),
                                 ("gamma", Dynamic.Mission.FLIGHT_PATH_ANGLE),
                                 ("i_wing", Aircraft.Wing.INCIDENCE)]

        elif alpha_mode is AlphaModes.DECELERATION:
            alpha_comp = om.BalanceComp(
                name="alpha",
                val=np.full(nn, 10),  # initial guess
                units="deg",
                lhs_name=Dynamic.Mission.VELOCITY_RATE,
                rhs_name='target_tas_rate',
                rhs_val=target_tas_rate,
                eq_units="kn/s",
                upper=25.,
                lower=-2.,
            )
            alpha_comp_inputs = [Dynamic.Mission.VELOCITY_RATE]

        elif alpha_mode is AlphaModes.REQUIRED_LIFT:
            alpha_comp = om.BalanceComp(
                name="alpha",
                val=8.0 * np.ones(nn),
                units="deg",
                rhs_name="required_lift",
                lhs_name=Dynamic.Mission.LIFT,
                eq_units="lbf",
                upper=12.0,
                lower=-2,
            )
            alpha_comp_inputs = ["required_lift", Dynamic.Mission.LIFT]

        # Future controller modes
        # elif alpha_mode is AlphaModes.FLIGHT_PATH_ANGLE:
        #     alpha_comp = om.BalanceComp(
        #         name="alpha",
        #         val=np.full(nn, 1),
        #         units="deg",
        #         lhs_name=Dynamic.Mission.FLIGHT_PATH_ANGLE,
        #         rhs_name='target_flight_path_angle',
        #         rhs_val=target_flight_path_angle,
        #         eq_units="deg",
        #         upper=12.0,
        #         lower=-2,
        #     )
        #     alpha_comp_inputs = [Dynamic.Mission.FLIGHT_PATH_ANGLE]

        # elif alpha_mode is AlphaModes.ALTITUDE_RATE:
        #     alpha_comp = om.BalanceComp(
        #         name="alpha",
        #         val=np.full(nn, 1),
        #         units="deg",
        #         lhs_name=Dynamic.Mission.ALTITUDE_RATE,
        #         rhs_name='target_alt_rate',
        #         rhs_val=target_alt_rate,
        #         upper=12.0,
        #         lower=-2,
        #     )
        #     alpha_comp_inputs = [Dynamic.Mission.ALTITUDE_RATE]

        # elif alpha_mode is AlphaModes.CONSTANT_ALTITUDE:
        #     alpha_comp = om.BalanceComp(
        #         name="alpha",
        #         val=np.full(nn, 1),
        #         units="deg",
        #         lhs_name=Dynamic.Mission.ALTITUDE,
        #         rhs_name='target_alt',
        #         rhs_val=37500,
        #         upper=12.0,
        #         lower=-2,
        #     )
        #     alpha_comp_inputs = [Dynamic.Mission.ALTITUDE]

        if alpha_mode is not AlphaModes.DEFAULT:
            alpha_group.add_subsystem("alpha_comp",
                                      alpha_comp,
                                      promotes_inputs=alpha_comp_inputs,
                                      promotes_outputs=["alpha"],
                                      )

            if add_default_solver and alpha_mode not in (AlphaModes.ROTATION,):
                alpha_group.nonlinear_solver = om.NewtonSolver()
                alpha_group.nonlinear_solver.options["solve_subsystems"] = True
                alpha_group.nonlinear_solver.options["iprint"] = print_level
                alpha_group.nonlinear_solver.options["atol"] = atol
                alpha_group.nonlinear_solver.options["rtol"] = rtol
                alpha_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
                alpha_group.linear_solver = om.DirectSolver(assemble_jac=True)

    def AddThrottleControl(
        self,
        prop_group=om.Group(),
        num_nodes=1,
        atol=1e-12,
        rtol=1e-12,
        add_default_solver=True,
        print_level=0,
    ):
        '''
        This is used when throttle in an ODE needs to be controlled directly.
        '''

        nn = num_nodes

        thrust_bal = om.BalanceComp(
            name=Dynamic.Mission.THROTTLE,
            val=np.ones(nn),
            upper=1.0,
            lower=0.0,
            units='unitless',
            lhs_name=Dynamic.Mission.THRUST_TOTAL,
            rhs_name="required_thrust",
            eq_units="lbf",
        )
        prop_group.add_subsystem("thrust_balance",
                                 thrust_bal,
                                 promotes_inputs=[
                                     Dynamic.Mission.THRUST_TOTAL, 'required_thrust'],
                                 promotes_outputs=[Dynamic.Mission.THROTTLE],
                                 )

        if add_default_solver:
            prop_group.linear_solver = om.DirectSolver()
            prop_group.linear_solver.options["iprint"] = print_level

            prop_group.nonlinear_solver = om.NewtonSolver()
            prop_group.nonlinear_solver.options["err_on_non_converge"] = False
            prop_group.nonlinear_solver.options["solve_subsystems"] = True
            prop_group.nonlinear_solver.options["maxiter"] = 20
            prop_group.nonlinear_solver.options["iprint"] = print_level
            prop_group.nonlinear_solver.options["atol"] = atol
            prop_group.nonlinear_solver.options["rtol"] = rtol
            prop_group.nonlinear_solver.linesearch = om.BoundsEnforceLS()
            prop_group.linear_solver = om.DirectSolver(assemble_jac=True)

        if prop_group is not self:
            self.add_subsystem(
                'prop_group',
                prop_group,
                promotes=['*']
            )

    def add_atmosphere(self, nn, input_speed_type=SpeedType.TAS):
        """Add atmosphere component"""
        self.add_subsystem(
            name='atmosphere',
            subsys=Atmosphere(num_nodes=nn, input_speed_type=input_speed_type),
            promotes=['*'],
        )

    def add_excess_rate_comps(self, nn):
        """Add SpecificEnergyRate and AltitudeRate components"""
        self.add_subsystem(
            name='SPECIFIC_ENERGY_RATE_EXCESS',
            subsys=SpecificEnergyRate(num_nodes=nn),
            promotes_inputs=[Dynamic.Mission.VELOCITY, Dynamic.Mission.MASS,
                             (Dynamic.Mission.THRUST_TOTAL, Dynamic.Mission.THRUST_MAX_TOTAL),
                             Dynamic.Mission.DRAG],
            promotes_outputs=[(Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                               Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS)]
        )

        self.add_subsystem(
            name='ALTITUDE_RATE_MAX',
            subsys=AltitudeRate(num_nodes=nn),
            promotes_inputs=[
                (Dynamic.Mission.SPECIFIC_ENERGY_RATE,
                 Dynamic.Mission.SPECIFIC_ENERGY_RATE_EXCESS),
                Dynamic.Mission.VELOCITY_RATE,
                Dynamic.Mission.VELOCITY],
            promotes_outputs=[
                (Dynamic.Mission.ALTITUDE_RATE,
                 Dynamic.Mission.ALTITUDE_RATE_MAX)])
