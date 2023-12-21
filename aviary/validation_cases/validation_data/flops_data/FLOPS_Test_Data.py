from aviary.models.large_single_aisle_2.large_single_aisle_2_altwt_FLOPS_data import \
    LargeSingleAisle2FLOPSalt
from aviary.models.large_single_aisle_2.large_single_aisle_2_detailwing_FLOPS_data import \
    LargeSingleAisle2FLOPSdw
from aviary.models.large_single_aisle_2.large_single_aisle_2_FLOPS_data import \
    LargeSingleAisle2FLOPS
from aviary.models.large_single_aisle_1.large_single_aisle_1_FLOPS_data import \
    LargeSingleAisle1FLOPS
from aviary.models.N3CC.N3CC_data import N3CC

FLOPS_Test_Data = {}

FLOPS_Test_Data['LargeSingleAisle1FLOPS'] = LargeSingleAisle1FLOPS
FLOPS_Test_Data['LargeSingleAisle2FLOPS'] = LargeSingleAisle2FLOPS
FLOPS_Test_Data['LargeSingleAisle2FLOPSdw'] = LargeSingleAisle2FLOPSdw
FLOPS_Test_Data['LargeSingleAisle2FLOPSalt'] = LargeSingleAisle2FLOPSalt
FLOPS_Test_Data['N3CC'] = N3CC
