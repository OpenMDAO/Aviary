from aviary.examples.external_subsystems.engine_NPSS.NPSS_variables import Dynamic

vars_to_connect = {
    'Fn_train': {
        'mission_name': [
            Dynamic.Vehicle.Propulsion.THRUST + '_train',
        ],
        'units': 'lbf',
    },
    'Fn_max_train': {
        'mission_name': [
            Dynamic.Vehicle.Propulsion.THRUST_MAX + '_train',
        ],
        'units': 'lbf',
    },
    'Wf_inv_train': {
        'mission_name': [
            Dynamic.Vehicle.Propulsion.FUEL_FLOW_RATE_NEGATIVE + '_train',
        ],
        'units': 'lbm/s',
    },
}
