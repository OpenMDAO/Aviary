import openmdao.api as om

from aviary.subsystems.aerodynamics.gasp_based.flaps_model.basic_calculations import \
    BasicFlapsCalculations
from aviary.subsystems.aerodynamics.gasp_based.flaps_model.Cl_max import \
    CLmaxCalculation
from aviary.subsystems.aerodynamics.gasp_based.flaps_model.L_and_D_increments import \
    LiftAndDragIncrements
from aviary.subsystems.aerodynamics.gasp_based.flaps_model.meta_model import \
    MetaModelGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import FlapType
from aviary.variable_info.variables import Aircraft, Dynamic


class FlapsGroup(om.Group):
    """
    Group connecting four components of the flaps model. They are: BasicFlapsCalculations,
    CLmaxCalculation, MetaModelGroup, and LiftAndDragIncrements. Then, a non-linear solver
    is provided.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

        # optimum trailing edge flap deflection angle defaults (ADELTO table in GASP)
        self.optimum_flap_defls = {
            FlapType.PLAIN: 60.0,
            FlapType.SPLIT: 60.0,
            FlapType.SINGLE_SLOTTED: 40.0,
            FlapType.DOUBLE_SLOTTED: 55.0,
            FlapType.TRIPLE_SLOTTED: 55.0,
            FlapType.FOWLER: 30.0,
            FlapType.DOUBLE_SLOTTED_FOWLER: 30.0,
        }

    def setup(self):

        aviary_options = self.options['aviary_options']

        self.add_subsystem(
            "BasicFlapsCalculations",
            BasicFlapsCalculations(),
            promotes_inputs=[
                "slat_defl",
                "flap_defl",
            ]
            + ["aircraft:*"],
            promotes_outputs=["*"],
        )

        self.add_subsystem(
            "CLmaxCalculation",
            CLmaxCalculation(),
            promotes_inputs=[
                Dynamic.Mission.SPEED_OF_SOUND,
                Dynamic.Mission.STATIC_PRESSURE,
                Dynamic.Mission.KINEMATIC_VISCOSITY,
                "VLAM1",
                "VLAM2",
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM8",
                "VLAM9",
                "VLAM10",
                "VLAM11",
                "VLAM12",
                "VLAM13",
                "VLAM14",
                "fus_lift",
                Dynamic.Mission.TEMPERATURE,
            ]
            + ["aircraft:*"],
            promotes_outputs=["CL_max", Dynamic.Mission.MACH, "reynolds"],
        )

        self.add_subsystem(
            "LookupTables",
            MetaModelGroup(aviary_options=aviary_options),
            promotes_inputs=[
                "flap_defl_ratio",
                "flap_defl",
                "slat_defl_ratio",
                "reynolds",
                Dynamic.Mission.MACH,
                "body_to_span_ratio",
                "chord_to_body_ratio",
            ]
            + ["aircraft:*"],
            promotes_outputs=[
                "VDEL1",
                "VDEL2",
                "VDEL3",
                "VLAM1",
                "VLAM2",
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM10",
                "VLAM11",
                "VLAM13",
                "VLAM14",
                "fus_lift",
            ],
        )

        self.add_subsystem(
            "LiftAndDragIncrements",
            LiftAndDragIncrements(),
            promotes_inputs=[
                Aircraft.Wing.FLAP_DRAG_INCREMENT_OPTIMUM,
                Aircraft.Wing.FLAP_LIFT_INCREMENT_OPTIMUM,
                "VDEL1",
                "VDEL2",
                "VDEL3",
                "VDEL4",
                "VDEL5",
                "VLAM3",
                "VLAM4",
                "VLAM5",
                "VLAM6",
                "VLAM7",
                "VLAM8",
                "VLAM13",
                "VLAM14",
            ],
            promotes_outputs=["delta_CD", "delta_CL"],
        )

        self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        self.linear_solver = om.DirectSolver()

        self.nonlinear_solver.options["iprint"] = 0
        self.nonlinear_solver.options["maxiter"] = 25
        self.nonlinear_solver.options["atol"] = 1e-8
        self.nonlinear_solver.options["rtol"] = 1e-8

        # set default trailing edge deflection angle per GASP
        self.set_input_defaults(
            Aircraft.Wing.OPTIMUM_FLAP_DEFLECTION,
            self.optimum_flap_defls[self.options["aviary_options"].get_val(
                Aircraft.Wing.FLAP_TYPE, units='unitless')],
            units="deg",
        )
