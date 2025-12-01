from aviary.models.aircraft.large_single_aisle_1.large_single_aisle_1_FLOPS_data import (
    LargeSingleAisle1FLOPS,
)
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_altwt_FLOPS_data import (
    LargeSingleAisle2FLOPSalt,
)
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_detailwing_FLOPS_data import (
    LargeSingleAisle2FLOPSdw,
)
from aviary.models.aircraft.large_single_aisle_2.large_single_aisle_2_FLOPS_data import (
    LargeSingleAisle2FLOPS,
)
from aviary.models.aircraft.multi_engine_single_aisle.multi_engine_single_aisle_data import (
    MultiEngineSingleAisle,
)
from aviary.models.aircraft.advanced_single_aisle.advanced_single_aisle_data import N3CC

FLOPS_Test_Data = {}

FLOPS_Test_Data['LargeSingleAisle1FLOPS'] = LargeSingleAisle1FLOPS
FLOPS_Test_Data['LargeSingleAisle2FLOPS'] = LargeSingleAisle2FLOPS
FLOPS_Test_Data['LargeSingleAisle2FLOPSdw'] = LargeSingleAisle2FLOPSdw
FLOPS_Test_Data['LargeSingleAisle2FLOPSalt'] = LargeSingleAisle2FLOPSalt
FLOPS_Test_Data['AdvancedSingleAisle'] = N3CC

# We don't have full date for this yet, but might still want to run one in a single unit test.
FLOPS_Lacking_Test_Data = {}
FLOPS_Lacking_Test_Data['MultiEngineSingleAisle'] = MultiEngineSingleAisle
