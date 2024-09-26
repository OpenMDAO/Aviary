import openmdao.api as om

from aviary.subsystems.geometry.flops_based.prep_geom import PrepGeom
from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.utils.aviary_values import AviaryValues
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft

FLOPS = LegacyCode.FLOPS
GASP = LegacyCode.GASP


class CombinedGeometry(om.Group):
    """
    Group that contains both FLOPS and GASP based pre-mission geometry components, for models that require both sets of geometry calculations.

    The "code_origin_to_prioritize" flag is used to determine which method's outputs should be used if both FLOPS and  GASP methods compute the same variable.
    """

    def initialize(self):
        self.options.declare(
            'aviary_options', types=AviaryValues,
            desc='collection of Aircraft/Mission specific options'
        )

        self.options.declare('code_origin_to_prioritize',
                             values=[GASP, FLOPS, None],
                             default=None,
                             desc='sets which code origin to prioritize if there are'
                                  ' conflicting outputs.'
                             )

    def setup(self):
        aviary_inputs = self.options['aviary_options']

        self.add_subsystem(
            'gasp_based_geom',
            SizeGroup(aviary_options=aviary_inputs,),
            promotes_inputs=["aircraft:*", "mission:*"],
            promotes_outputs=["aircraft:*"],
        )

        self.add_subsystem(
            'flops_based_geom',
            PrepGeom(aviary_options=aviary_inputs),
            promotes_inputs=['*'],
            promotes_outputs=['*']
        )

    def configure(self):
        prioritize_origin = self.options['code_origin_to_prioritize']
        override = self.manual_overrides = None

        # These are outputs that are computed by both flops_based and gasp_based
        # geometry subsystems.
        flops_geom_pathname = self.flops_based_geom.pathname
        flops_fus_area_path = flops_geom_pathname + '.fuselage.' + Aircraft.Fuselage.WETTED_AREA
        flops_fus_diam_path = flops_geom_pathname + \
            '.fuselage_prelim.' + Aircraft.Fuselage.AVG_DIAMETER
        gasp_geom_pathname = self.gasp_based_geom.pathname
        gasp_fus_area_path = gasp_geom_pathname + '.fuselage.size.' + Aircraft.Fuselage.WETTED_AREA
        gasp_fus_diam_path = gasp_geom_pathname + \
            '.fuselage.parameters.' + Aircraft.Fuselage.AVG_DIAMETER

        if prioritize_origin is GASP:
            override = [flops_fus_area_path, flops_fus_diam_path]

            name = Aircraft.Fuselage.WETTED_AREA
            outs = [(name, f"MANUAL_OVERRIDE:{name}")]
            self.flops_based_geom.promotes('fuselage', outputs=outs)

            name = Aircraft.Fuselage.AVG_DIAMETER
            outs = [(name, f"MANUAL_OVERRIDE:{name}")]
            self.flops_based_geom.promotes('fuselage_prelim', outputs=outs)

        elif prioritize_origin is FLOPS:
            override = [gasp_fus_area_path, gasp_fus_diam_path]

            name = Aircraft.Fuselage.WETTED_AREA
            outs = [(name, f"MANUAL_OVERRIDE:{name}")]
            self.gasp_based_geom.fuselage.promotes('size', outputs=outs)

            name = Aircraft.Fuselage.AVG_DIAMETER
            outs = [(name, f"MANUAL_OVERRIDE:{name}")]
            self.gasp_based_geom.fuselage.promotes('parameters', outputs=outs)

        self.manual_overrides = override
