import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials
from aviary.models.external_subsystems.UAV.aerodynamics.model.aero_model import (
    WingTailAreaRatios,
    FuselageDrag,
    VTailDrag,
    LandingGearDrag,
    Averages,
    TotalAircraftAero,
)
from aviary.models.external_subsystems.UAV.UAV_variable_info.UAV_variables import Dynamic, Aircraft


class TestWingTailAreaRatios(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', WingTailAreaRatios(num_nodes=3), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.SPAN, val=3.0)
        prob.set_val(Aircraft.Wing.ROOT_CHORD, val=4.0)
        prob.set_val(Aircraft.HorizontalTail.SPAN, val=5.0)
        prob.set_val(Aircraft.HorizontalTail.ROOT_CHORD, val=6.0)
        prob.set_val(Aircraft.VerticalTail.SPAN, val=7.0)
        prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, val=8.0)

        prob.run_model()

        cp_data = prob.check_partials(compact_print=True, out_stream=None, method='fd')
        
        assert_check_partials(cp_data, atol=1e-6, rtol=1e-6)

class TestFuselageDrag(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', FuselageDrag(num_nodes=3), promotes=['*'])
        
        # Still needed because your declare_partials uses method='cs'
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, val=1.2)
        prob.set_val('Cf_fus', val=0.0044)
        prob.set_val('CD_L_fus', val=0.001)
        
        prob.set_val(Aircraft.Fuselage.LENGTH, val=10.0)
        prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, val=2.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=1.5)
        prob.set_val(Aircraft.Wing.AREA, val=15.0)
        
        prob.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, val=np.array([50000.0, 75000.0, 101325.0])) 

        prob.run_model()

        cp_data = prob.check_partials(compact_print=True, out_stream=None, method='fd', form='central')
        
        assert_check_partials(cp_data, atol=1e-5, rtol=1e-5)

class TestVTailDrag(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', VTailDrag(num_nodes=3), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        # Set distinct realistic scalar values to prevent 1.0 or 0.0 from hiding math errors
        prob.set_val('R_LS', val=1.1)
        prob.set_val('Cf_vtail', val=0.0044)
        prob.set_val('L_prime', val=1.2)
        
        prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, val=1.5)
        prob.set_val(Aircraft.VerticalTail.TAPER_RATIO, val=0.5)
        prob.set_val(Aircraft.VerticalTail.SPAN, val=4.0)
        prob.set_val(Aircraft.Wing.AREA, val=15.0)
        prob.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, val=0.12)
        
        prob.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, val=np.array([50000.0, 75000.0, 101325.0])) 

        prob.run_model()

        # Check partials using central finite difference 
        cp_data = prob.check_partials(
            compact_print=True, 
            out_stream=None, 
            method='fd', 
            form='central'
        )
        
        # Assert derivatives match within tolerances
        assert_check_partials(cp_data, atol=1e-6, rtol=1e-6)

class TestLandingGearDrag(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', LandingGearDrag(num_nodes=3), promotes=['*'])
        prob.setup(force_alloc_complex=True)

        prob.set_val(Aircraft.LandingGear.DRAG_COEFFICIENT, val=0.05)
        prob.set_val(Aircraft.Wing.AREA, val=15.0)
        prob.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, val=np.array([50000.0, 75000.0, 101325.0])) 

        prob.run_model()

        cp_data = prob.check_partials(
            compact_print=True, 
            out_stream=None, 
            method='cs', 
        )
        
        assert_check_partials(cp_data, atol=1e-6, rtol=1e-6)

class TestAverages(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', Averages(num_nodes=4), promotes=['*'])
        
        prob.setup(force_alloc_complex=True)

        # Provide distinct arrays so we know it's averaging actual data
        prob.set_val('CD', val=np.array([0.02, 0.025, 0.03, 0.035]))
        prob.set_val('CD_fus', val=np.array([0.005, 0.006, 0.007, 0.008]))
        prob.set_val('lifting_surface_CL', val=np.array([0.3, 0.5, 0.8, 1.2]))

        prob.run_model()

        # Check partials using central finite difference
        cp_data = prob.check_partials(
            compact_print=True, 
            out_stream=None, 
            method='fd', 
            form='central'
        )
        
        # Because the math is purely linear, FD will match analytical exactly. 
        # 1e-6 tolerances are more than sufficient.
        assert_check_partials(cp_data, atol=1e-6, rtol=1e-6)

# TODO: There should be a avlues test for TotalAircraftAero, and values tests for all the other tests as well to sanity check the results.

if __name__ == '__main__':
    unittest.main()
