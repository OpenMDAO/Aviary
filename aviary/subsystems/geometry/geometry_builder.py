"""
Define subsystem builder for Aviary core geometry.

Classes
-------
GeometryBuilderBase: the interface for a geometry subsystem builder.

CoreGeometryBuilder : the interface for Aviary's core geometry subsystem builder
"""

from aviary.interface.utils.markdown_utils import write_markdown_variable_table
from aviary.subsystems.geometry.combined_geometry import CombinedGeometry
from aviary.subsystems.geometry.flops_based.prep_geom import PrepGeom
from aviary.subsystems.geometry.gasp_based.size_group import SizeGroup
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft

GASP = LegacyCode.GASP
FLOPS = LegacyCode.FLOPS

_default_name = 'geometry'


class GeometryBuilderBase(SubsystemBuilderBase):
    """
    Base geometry builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None):
        Initializes the GeometryBuilderBase object with a given name.
    mission_inputs(self, **kwargs) -> list:
        Return mission inputs.
    mission_outputs(self, **kwargs) -> list:
        Return mission outputs.
    """

    def __init__(self, name=None, meta_data=None):
        if name is None:
            name = _default_name

        super().__init__(name=name, meta_data=meta_data)

    def mission_inputs(self, **kwargs):
        return ['*']

    def mission_outputs(self, **kwargs):
        return ['*']


class CoreGeometryBuilder(GeometryBuilderBase):
    """
    Core geometry builder.

    Methods
    -------
    __init__(self, name=None, meta_data=None, code_origin=None,
        code_origin_to_prioritize=None):
    build_pre_mission(self, aviary_inputs) -> openmdao.core.System:
        Builds an OpenMDAO system for the pre-mission computations of the subsystem.
    build_mission(self, num_nodes, aviary_inputs, **kwargs) -> openmdao.core.System:
        Builds an OpenMDAO system for the mission computations of the subsystem.
    get_parameters(self, aviary_inputs=None, phase_info=None):
        Returns a dictionary of fixed values for the Nacelle.
    report(self, prob, reports_folder, **kwargs):
        Generate the report for Aviary core geometry analysis.
    """

    def __init__(
        self,
        name=None,
        meta_data=None,
        code_origin=None,
        code_origin_to_prioritize=None,
    ):
        if name is None:
            name = 'core_geometry'

        if code_origin not in (FLOPS, GASP) and set(code_origin) != set((FLOPS, GASP)):
            raise ValueError('Code origin is not one of the following: (FLOPS, GASP)')

        self.code_origin = code_origin
        self.use_both_geometries = code_origin == (FLOPS, GASP)
        self.code_origin_to_prioritize = code_origin_to_prioritize

        super().__init__(name=name, meta_data=meta_data)

    def build_pre_mission(self, aviary_inputs, **kwargs):
        code_origin = self.code_origin
        both_geom = self.use_both_geometries
        code_origin_to_prioritize = self.code_origin_to_prioritize
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None

        geom_group = None

        if method != 'external':
            if both_geom:
                geom_group = CombinedGeometry(code_origin_to_prioritize=code_origin_to_prioritize)

            elif code_origin is GASP:
                geom_group = SizeGroup()
                geom_group.manual_overrides = None

            elif code_origin is FLOPS:
                geom_group = PrepGeom()
                geom_group.manual_overrides = None

        return geom_group

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        # by default there is no geom mission, but call super for safety/future-proofing
        try:
            method = kwargs.pop('method')
        except KeyError:
            method = None
        geom_group = None

        if method != 'external':
            geom_group = super().build_mission(num_nodes, aviary_inputs)

        return geom_group

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        num_engine_type = len(aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES))
        params = {}

        for entry in Aircraft.Nacelle.__dict__:
            if entry != '__dict__':  # cannot get attribute from mappingproxy
                var = getattr(Aircraft.Nacelle, entry)
                if var in aviary_inputs:
                    if 'total' not in var:
                        params[var] = {
                            'shape': (num_engine_type),
                            'static_target': True,
                        }

        return params

    def report(self, prob, reports_folder, **kwargs):
        """
        Generate the report for Aviary core geometry analysis.

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
        wing_outputs = [
            Aircraft.Wing.AREA,
            Aircraft.Wing.SPAN,
            Aircraft.Wing.ASPECT_RATIO,
            Aircraft.Wing.SWEEP,
        ]
        htail_outputs = [Aircraft.HorizontalTail.AREA, Aircraft.VerticalTail.AREA]
        fuselage_outputs = [Aircraft.Fuselage.LENGTH, Aircraft.Fuselage.AVG_DIAMETER]

        with open(filepath, mode='w') as f:
            if self.use_both_geometries:
                method = 'FLOPS and GASP methods'
            else:
                method = self.code_origin.value + ' method'
            f.write(f'# Geometry: {method}\n')
            f.write('## Wing')
            write_markdown_variable_table(f, prob, wing_outputs, self.meta_data)
            f.write('\n## Empennage\n')
            f.write('### Horizontal Tail')
            write_markdown_variable_table(f, prob, htail_outputs, self.meta_data)
            f.write('\n## Fuselage')
            write_markdown_variable_table(f, prob, fuselage_outputs, self.meta_data)
