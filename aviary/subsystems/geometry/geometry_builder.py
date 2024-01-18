"""
Define subsystem builder for Aviary core geometry.

Classes
-------
GeometryBuilderBase: the interface for a geometry subsystem builder.

CoreGeometryBuilder : the interface for Aviary's core geometry subsystem builder
"""

from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.subsystems.geometry.combined_geometry import CombinedGeometry
from aviary.subsystems.geometry.flops_based.prep_geom import PrepGeom
from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.enums import LegacyCode


GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS

_default_name = 'geometry'


class GeometryBuilderBase(SubsystemBuilderBase):
    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CoreGeometryBuilder(GeometryBuilderBase):
    def __init__(self, name=None, meta_data=None, code_origin=None,
                 use_both_geometries=False, code_origin_to_prioritize=None):
        if name is None:
            name = 'core_geometry'

        if code_origin not in (FLOPS, GASP) and not use_both_geometries:
            raise ValueError('Code origin is not one of the following: (FLOPS, GASP)')

        self.code_origin = code_origin
        self.use_both_geometries = use_both_geometries
        self.code_origin_to_prioritize = code_origin_to_prioritize

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs):
        code_origin = self.code_origin
        both_geom = self.use_both_geometries
        code_origin_to_prioritize = self.code_origin_to_prioritize

        geom_group = None

        if both_geom:
            geom_group = CombinedGeometry(aviary_options=aviary_inputs,
                                          code_origin_to_prioritize=code_origin_to_prioritize)

        elif code_origin is GASP:
            geom_group = SizeGroup(aviary_options=aviary_inputs)
            geom_group.manual_overrides = None

        elif code_origin is FLOPS:
            geom_group = PrepGeom(aviary_options=aviary_inputs)
            geom_group.manual_overrides = None

        return geom_group

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        super().build_mission(num_nodes, aviary_inputs)

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core geometry analysis

        Parameters
        ----------
        prob : AviaryProblem
            The AviaryProblem that will be used to generate the report
        reports_folder : Path
            Location of the subsystems_report folder this report will be placed in
        """
        filename = self.name + '.md'
        filepath = reports_folder / filename

        # TODO output differs by method
        # TODO finish variables of interest
        wing_outputs = [Aircraft.Wing.AREA,
                        Aircraft.Wing.SPAN,
                        Aircraft.Wing.ASPECT_RATIO,
                        Aircraft.Wing.SWEEP]
        htail_outputs = [Aircraft.HorizontalTail.AREA,
                         Aircraft.VerticalTail.AREA]
        fuselage_outputs = [Aircraft.Fuselage.LENGTH,
                            Aircraft.Fuselage.AVG_DIAMETER]

        with open(filepath, mode='w') as f:
            method = self.code_origin + ' METHOD'
            if self.use_both_geometries:
                method = ('FLOPS AND GASP METHODS')
            f.write(f'# GEOMETRY: {method}\n')
            f.write('## Wing')
            write_markdown_variable_table(f, prob, wing_outputs, self.meta_data)
            f.write('\n## Empennage\n')
            f.write('### Horizontal Tail')
            write_markdown_variable_table(f, prob, htail_outputs, self.meta_data)
            f.write('\n## Fuselage')
            write_markdown_variable_table(f, prob, fuselage_outputs, self.meta_data)
