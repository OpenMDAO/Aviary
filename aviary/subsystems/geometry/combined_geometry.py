import openmdao.api as om

from aviary.subsystems.geometry.flops_based.prep_geom import PrepGeom
from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
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
            'code_origin_to_prioritize',
            values=[GASP, FLOPS, None],
            default=None,
            desc='sets which code origin to prioritize if there are conflicting outputs.',
        )

    def setup(self):
        self.add_subsystem(
            'gasp_based_geom',
            SizeGroup(),
            promotes_inputs=['aircraft:*'],
            promotes_outputs=['aircraft:*'],
        )

        self.add_subsystem(
            'flops_based_geom', PrepGeom(), promotes_inputs=['*'], promotes_outputs=['*']
        )

    def configure(self):
        prioritize_origin = self.options['code_origin_to_prioritize']
        override = self.code_origin_overrides = None

        # These are outputs that are computed by both flops_based and gasp_based geometry subsystems.
        flops_geom_pathname = self.flops_based_geom.pathname
        flops_fus_area_path = (
            flops_geom_pathname + '.wetted_area.fus_swet.' + Aircraft.Fuselage.WETTED_AREA
        )
        flops_nac_diam_path = (
            flops_geom_pathname + '.nacelle_characteristic_lengths.' + Aircraft.Nacelle.AVG_DIAMETER
        )
        flops_nac_len_path = (
            flops_geom_pathname + '.nacelle_characteristic_lengths.' + Aircraft.Nacelle.AVG_LENGTH
        )

        gasp_geom_pathname = self.gasp_based_geom.pathname
        gasp_fus_area_path = gasp_geom_pathname + '.fuselage.size.' + Aircraft.Fuselage.WETTED_AREA
        gasp_nac_diam_path = (
            gasp_geom_pathname + '.engine.eng_diameter.' + Aircraft.Nacelle.AVG_DIAMETER
        )
        gasp_nac_len_path = gasp_geom_pathname + '.engine.eng_length.' + Aircraft.Nacelle.AVG_LENGTH

        # Pre-fetch the names for cleaner code
        fus_area_name = Aircraft.Fuselage.WETTED_AREA
        nac_diam_name = Aircraft.Nacelle.AVG_DIAMETER
        nac_len_name = Aircraft.Nacelle.AVG_LENGTH

        if prioritize_origin is GASP:
            # Add all FLOPS overridden variables to the list
            override = [flops_fus_area_path, flops_nac_diam_path, flops_nac_len_path]

            fus_outs = [(fus_area_name, f'CODE_ORIGIN_OVERRIDE:{fus_area_name}')]
            self.flops_based_geom.promotes('wetted_area', outputs=fus_outs)

            # Override the FLOPS nacelle length and diameter
            nac_outs = [
                (nac_diam_name, f'CODE_ORIGIN_OVERRIDE:{nac_diam_name}'),
                (nac_len_name, f'CODE_ORIGIN_OVERRIDE:{nac_len_name}'),
            ]
            self.flops_based_geom.promotes('nacelle_characteristic_lengths', outputs=nac_outs)

        elif prioritize_origin is FLOPS:
            # Add all GASP overridden variables to the list
            override = [gasp_fus_area_path, gasp_nac_diam_path, gasp_nac_len_path]

            fus_outs = [(fus_area_name, f'CODE_ORIGIN_OVERRIDE:{fus_area_name}')]
            self.gasp_based_geom.fuselage.promotes('size', outputs=fus_outs)

            # Override the GASP nacelle length and diameter
            nac_diam_out = [(nac_diam_name, f'CODE_ORIGIN_OVERRIDE:{nac_diam_name}')]
            nac_len_out = [(nac_len_name, f'CODE_ORIGIN_OVERRIDE:{nac_len_name}')]
            # Since eng_diameter is a component of engine in GASP, promote from engine
            self.gasp_based_geom.engine.promotes('eng_diameter', outputs=nac_diam_out)
            self.gasp_based_geom.engine.promotes('eng_length', outputs=nac_len_out)

        self.code_origin_overrides = override
