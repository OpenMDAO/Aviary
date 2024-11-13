"""
OpenMDAO System to compute drag based on the methods in FLOPS AERO.
"""
import numpy as np
import openmdao.api as om

from aviary.subsystems.aerodynamics.aero_common import DynamicPressure
from aviary.subsystems.aerodynamics.flops_based.buffet_lift import BuffetLift
from aviary.subsystems.aerodynamics.flops_based.compressibility_drag import \
    CompressibilityDrag
from aviary.subsystems.aerodynamics.flops_based.drag import TotalDrag
from aviary.subsystems.aerodynamics.flops_based.induced_drag import InducedDrag
from aviary.subsystems.aerodynamics.flops_based.lift import LiftEqualsWeight
from aviary.subsystems.aerodynamics.flops_based.lift_dependent_drag import \
    LiftDependentDrag
from aviary.subsystems.aerodynamics.flops_based.mux_component import MuxComponent
from aviary.subsystems.aerodynamics.flops_based.skin_friction import SkinFriction
from aviary.subsystems.aerodynamics.flops_based.skin_friction_drag import \
    SkinFrictionDrag
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


class ComputedAeroGroup(om.Group):
    """
    FLOPS-based computed aero group
    """

    def initialize(self):
        self.options.declare(
            "num_nodes", default=1, types=int,
            desc="Number of nodes along mission segment")
        self.options.declare(
            'gamma', default=1.4,
            desc='Ratio of specific heats for air.')
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options')

    def setup(self):
        num_nodes = self.options["num_nodes"]
        gamma = self.options['gamma']
        aviary_options: AviaryValues = self.options['aviary_options']

        comp = MuxComponent(aviary_options=aviary_options)
        self.add_subsystem(
            'Mux', comp,
            promotes_inputs=['aircraft:*'],
            promotes_outputs=[
                'wetted_areas', 'fineness_ratios', 'characteristic_lengths',
                'laminar_fractions_upper', 'laminar_fractions_lower'])

        self.add_subsystem(
            'DynamicPressure', DynamicPressure(num_nodes=num_nodes, gamma=gamma),
            promotes_inputs=[Dynamic.Mission.MACH, Dynamic.Mission.STATIC_PRESSURE],
            promotes_outputs=[Dynamic.Mission.DYNAMIC_PRESSURE])

        comp = LiftEqualsWeight(num_nodes=num_nodes)
        self.add_subsystem(
            name=Dynamic.Mission.LIFT, subsys=comp,
            promotes_inputs=[Aircraft.Wing.AREA, Dynamic.Mission.MASS,
                             Dynamic.Mission.DYNAMIC_PRESSURE],
            promotes_outputs=['cl', Dynamic.Mission.LIFT])

        comp = LiftDependentDrag(num_nodes=num_nodes, gamma=gamma)
        self.add_subsystem(
            'PressureDrag', comp,
            promotes_inputs=[
                Dynamic.Mission.MACH, Dynamic.Mission.LIFT, Dynamic.Mission.STATIC_PRESSURE,
                Mission.Design.MACH,
                Mission.Design.LIFT_COEFFICIENT,
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.THICKNESS_TO_CHORD])

        comp = InducedDrag(
            num_nodes=num_nodes, gamma=gamma, aviary_options=aviary_options)
        self.add_subsystem(
            'InducedDrag', comp,
            promotes_inputs=[
                Dynamic.Mission.MACH, Dynamic.Mission.LIFT, Dynamic.Mission.STATIC_PRESSURE,
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.SPAN_EFFICIENCY_FACTOR,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO])

        comp = CompressibilityDrag(num_nodes=num_nodes)
        self.add_subsystem(
            'CompressibilityDrag', comp,
            promotes_inputs=[
                Dynamic.Mission.MACH,
                Mission.Design.MACH,
                Aircraft.Design.BASE_AREA,
                Aircraft.Wing.AREA,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.TAPER_RATIO,
                Aircraft.Wing.THICKNESS_TO_CHORD,
                Aircraft.Fuselage.CROSS_SECTION,
                Aircraft.Fuselage.DIAMETER_TO_WING_SPAN,
                Aircraft.Fuselage.LENGTH_TO_DIAMETER])

        comp = SkinFriction(num_nodes=num_nodes, aviary_options=aviary_options)
        self.add_subsystem(
            'SkinFrictionCoef', comp,
            promotes_inputs=[
                Dynamic.Mission.MACH, Dynamic.Mission.STATIC_PRESSURE, Dynamic.Mission.TEMPERATURE,
                'characteristic_lengths'],
            promotes_outputs=['skin_friction_coeff', 'Re'])

        comp = SkinFrictionDrag(num_nodes=num_nodes, aviary_options=aviary_options)
        self.add_subsystem(
            'SkinFrictionDrag', comp,
            promotes_inputs=[
                'skin_friction_coeff', 'Re', 'fineness_ratios', 'wetted_areas',
                'laminar_fractions_upper', 'laminar_fractions_lower',
                Aircraft.Wing.AREA])

        comp = ComputedDrag(num_nodes=num_nodes)
        self.add_subsystem(
            'Drag', comp,
            promotes_inputs=[
                Dynamic.Mission.DYNAMIC_PRESSURE, Dynamic.Mission.MACH, Aircraft.Wing.AREA,
                Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
                Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR],
            promotes_outputs=['CDI', 'CD0', 'CD', Dynamic.Mission.DRAG])

        buf = BuffetLift(num_nodes=num_nodes)
        self.add_subsystem(
            'Buffet', buf,
            promotes_inputs=[
                Dynamic.Mission.MACH,
                Mission.Design.MACH,
                Aircraft.Wing.ASPECT_RATIO,
                Aircraft.Wing.MAX_CAMBER_AT_70_SEMISPAN,
                Aircraft.Wing.SWEEP,
                Aircraft.Wing.THICKNESS_TO_CHORD])

        self.connect('PressureDrag.CD', 'Drag.pressure_drag_coeff')
        self.connect('InducedDrag.induced_drag_coeff', 'Drag.induced_drag_coeff')
        self.connect(
            'CompressibilityDrag.compress_drag_coeff', 'Drag.compress_drag_coeff')
        self.connect(
            'SkinFrictionDrag.skin_friction_drag_coeff', 'Drag.skin_friction_drag_coeff')


class ComputedDrag(om.Group):
    """
    FLOPS-based computed drag group
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options["num_nodes"]

        self._setup_drag_coeff(
            'CDI',
            input0='pressure_drag_coeff', input1='induced_drag_coeff', output='CDI',
            desc='lift-dependent drag coefficient,'
            ' including contributions from pressure drag coefficient')

        self._setup_drag_coeff(
            'CD0',
            input0='skin_friction_drag_coeff', input1='compress_drag_coeff',
            output='CD0',
            desc='zero-lift drag coefficient')

        self.add_subsystem(
            Dynamic.Mission.DRAG, TotalDrag(num_nodes=nn),
            promotes_inputs=[
                Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR,
                Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR,
                Aircraft.Wing.AREA,
                Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR,
                Aircraft.Design.SUPERSONIC_DRAG_COEFF_FACTOR,
                'CDI', 'CD0', Dynamic.Mission.MACH, Dynamic.Mission.DYNAMIC_PRESSURE],
            promotes_outputs=['CD', Dynamic.Mission.DRAG])

        self.set_input_defaults(Aircraft.Wing.AREA, 1., 'ft**2')

    def _setup_drag_coeff(self, name, input0, input1, output, desc=None):
        nn = self.options["num_nodes"]

        input_args = {'val': np.ones(nn), 'units': 'unitless'}
        output_args = dict(input_args)

        if desc is not None:
            output_args['desc'] = desc

        kwargs = {input0: input_args, input1: input_args, output: output_args}

        subsys = self.add_subsystem(
            name, om.ExecComp(f'{output} = {input0} + {input1}', **kwargs),
            promotes_inputs=[input0, input1], promotes_outputs=[output])
        subsys.declare_coloring(show_summary=False)
