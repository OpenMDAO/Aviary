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
        
        # 7. Assert that the derivatives match within tolerances
        assert_check_partials(cp_data, atol=1e-6, rtol=1e-6)

class TestFuselageDrag(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        prob.model.add_subsystem('comp', FuselageDrag(num_nodes=3), promotes=['*'])
        
        # Still needed because your declare_partials uses method='cs'
        prob.setup(force_alloc_complex=True)

        # Set realistic positive scalar values to avoid division by zero 
        # (e.g., dividing by df or S_fus)
        prob.set_val(Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, val=1.2)
        prob.set_val('Cf_fus', val=0.0044)
        prob.set_val('CD_L_fus', val=0.001)
        
        prob.set_val(Aircraft.Fuselage.LENGTH, val=10.0)
        prob.set_val(Aircraft.Fuselage.MAX_HEIGHT, val=2.0)
        prob.set_val(Aircraft.Fuselage.MAX_WIDTH, val=1.5)
        prob.set_val(Aircraft.Wing.AREA, val=15.0)
        
        # Dynamic pressure is vectorized in your setup, but passing a scalar 
        # here will safely broadcast to all 3 nodes
        prob.set_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, val=np.array([50000.0, 75000.0, 101325.0])) 

        prob.run_model()

        # Check partials using finite difference ('fd')
        cp_data = prob.check_partials(compact_print=True, out_stream=None, method='fd', form='central')
        
        # Note: I loosened the tolerance slightly to 1e-5. 
        # Because the drag equations are highly non-linear (involving cubes and square roots), 
        # Finite Difference is prone to truncation error here. 
        # If it fails at 1e-6, 1e-5 is completely acceptable for FD checks.
        assert_check_partials(cp_data, atol=1e-5, rtol=1e-5)

class TestVTailDrag(unittest.TestCase):
    def test_partials(self):
        prob = om.Problem()
        nn = 3
        
        # Add the component (assuming VTailDrag is imported or defined above)
        prob.model.add_subsystem('comp', VTailDrag(num_nodes=nn), promotes=['*'])
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
        
        # Set dynamic pressure as a vector to ensure correct vectorized behavior
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

if __name__ == '__main__':
    unittest.main()
