from aviary.examples.external_subsystems.engine_NPSS.engine_variables import Aircraft, Dynamic

vars_to_connect = {
    "Fn_train" : {
        "mission_name": [
            Dynamic.Mission.THRUST+"_train",
        ],
        "units": "lbf",
    },
    "Fn_max_train": {
        "mission_name": [
            Dynamic.Mission.THRUST_MAX+"_train",
        ],
        "units": "lbf",
    },
    "Wf_inv_train": {
        "mission_name": [
            Dynamic.Mission.FUEL_FLOW_RATE_NEGATIVE+"_train",
        ],
        "units": "lbm/s",
    },
}
