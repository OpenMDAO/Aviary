import unittest

import numpy as np
import openmdao.api as om

from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.geometry.geometry_builder import CoreGeometryBuilder
from aviary.variable_info.enums import LegacyCode
from aviary.variable_info.variables import Aircraft
from aviary.variable_info.variable_meta_data import _MetaData as BaseMetaData
import aviary.api as av

GASP = LegacyCode.GASP


class TestGASPGeomBuilder(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreGeometryBuilder(
            'core_geometry',
            BaseMetaData,
            use_both_geometries=False,
            code_origin=GASP,
            code_origin_to_prioritize=GASP)
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(
            Aircraft.Electrical.HAS_HYBRID_SYSTEM, False, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_FOLD, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_STRUT, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Wing.CHOOSE_FOLD_LOCATION, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, True, units='unitless')


class TestGASPGeomBuilderHybrid(av.TestSubsystemBuilderBase):
    """
    That class inherits from TestSubsystemBuilder. So all the test functions are
    within that inherited class. The setUp() method prepares the class and is run
    before the test methods; then the test methods are run.
    """

    def setUp(self):
        self.subsystem_builder = CoreGeometryBuilder(
            'core_geometry',
            BaseMetaData,
            use_both_geometries=False,
            code_origin=GASP,
            code_origin_to_prioritize=GASP)
        self.aviary_values = av.AviaryValues()
        self.aviary_values.set_val(Aircraft.Engine.NUM_ENGINES, [1], units='unitless')
        self.aviary_values.set_val(
            Aircraft.Electrical.HAS_HYBRID_SYSTEM, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_FOLD, True, units='unitless')
        self.aviary_values.set_val(Aircraft.Wing.HAS_STRUT, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Design.COMPUTE_HTAIL_VOLUME_COEFF, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Design.COMPUTE_VTAIL_VOLUME_COEFF, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Wing.SPAN_EFFICIENCY_REDUCTION, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Wing.CHOOSE_FOLD_LOCATION, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Wing.FOLD_DIMENSIONAL_LOCATION_SPECIFIED, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Strut.DIMENSIONAL_LOCATION_SPECIFIED, True, units='unitless')
        self.aviary_values.set_val(
            Aircraft.Propulsion.TOTAL_NUM_WING_ENGINES, 2, units='unitless')


if __name__ == '__main__':
    unittest.main()
