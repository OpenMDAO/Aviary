"""
Define an aviary mission with an NPSS defined engine. During pre-mission the engine is designed and
an engine deck is made. During the mission the deck is used for performance. Weight is estimated
using the default Aviary method. The engine model was developed using NPSS v3.2.
"""

import aviary.api as av
from aviary.examples.external_subsystems.engine_NPSS.NPSS_engine_builder import (
    NPSSTabularEngineBuilder,
)
from aviary.examples.external_subsystems.engine_NPSS.NPSS_variable_meta_data import ExtendedMetaData

"""Build NPSS model in Aviary."""
phase_info = av.default_height_energy_phase_info

prob = av.AviaryProblem()

prob.options['group_by_pre_opt_post'] = True

# Load aircraft and options data from user
# Allow for user overrides here
# add engine builder
prob.load_inputs(
    'models/aircraft/advanced_single_aisle/advanced_single_aisle_FLOPS.csv',
    phase_info,
    engine_builders=[NPSSTabularEngineBuilder()],
    meta_data=ExtendedMetaData,
)

prob.check_and_preprocess_inputs()

prob.build_model()

prob.add_driver('SLSQP')

prob.add_design_variables()

prob.add_objective()

prob.setup()

prob.run_aviary_problem(suppress_solver_print=True)
