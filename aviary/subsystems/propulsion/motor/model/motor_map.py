import numpy as np

import openmdao.api as om

from aviary.variable_info.variables import Dynamic, Aircraft


motor_map = np.array(
    [
        # speed---- .0       .083333  .16667  .25    .33333.41667  .5,     .58333 .66667  .75,   .83333, .91667 1.
        [
            0.871,
            0.872,
            0.862,
            0.853,
            0.845,
            0.838,
            0.832,
            0.825,
            0.819,
            0.813,
            0.807,
            0.802,
            0.796,
        ],  # 0
        [
            0.872,
            0.873,
            0.863,
            0.854,
            0.846,
            0.839,
            0.833,
            0.826,
            0.820,
            0.814,
            0.808,
            0.803,
            0.797,
        ],  # 0.040
        [
            0.928,
            0.928,
            0.932,
            0.930,
            0.928,
            0.926,
            0.923,
            0.920,
            0.918,
            0.915,
            0.912,
            0.909,
            0.907,
        ],  # 0.104
        [
            0.931,
            0.932,
            0.944,
            0.947,
            0.947,
            0.947,
            0.946,
            0.945,
            0.943,
            0.942,
            0.940,
            0.938,
            0.937,
        ],  # 0.168
        [
            0.931,
            0.927,
            0.946,
            0.952,
            0.954,
            0.955,
            0.955,
            0.954,
            0.954,
            0.953,
            0.952,
            0.951,
            0.950,
        ],  # 0.232
        [
            0.917,
            0.918,
            0.944,
            0.952,
            0.956,
            0.958,
            0.958,
            0.959,
            0.959,
            0.958,
            0.958,
            0.957,
            0.956,
        ],  # 0.296
        [
            0.907,
            0.908,
            0.940,
            0.951,
            0.956,
            0.958,
            0.960,
            0.961,
            0.961,
            0.961,
            0.961,
            0.960,
            0.960,
        ],  # 0.360
        [
            0.897,
            0.898,
            0.935,
            0.948,
            0.954,
            0.958,
            0.960,
            0.961,
            0.962,
            0.962,
            0.962,
            0.962,
            0.962,
        ],  # 0.424
        [
            0.886,
            0.887,
            0.930,
            0.945,
            0.952,
            0.956,
            0.959,
            0.960,
            0.962,
            0.962,
            0.963,
            0.963,
            0.963,
        ],  # 0.488
        [
            0.875,
            0.876,
            0.924,
            0.941,
            0.949,
            0.954,
            0.957,
            0.960,
            0.961,
            0.962,
            0.962,
            0.963,
            0.963,
        ],  # 0.552
        [
            0.864,
            0.865,
            0.918,
            0.937,
            0.946,
            0.952,
            0.956,
            0.958,
            0.960,
            0.961,
            0.962,
            0.962,
            0.962,
        ],  # 0.616
        [
            0.853,
            0.854,
            0.912,
            0.932,
            0.943,
            0.949,
            0.953,
            0.956,
            0.958,
            0.960,
            0.961,
            0.961,
            0.962,
        ],  # 0.680
        [
            0.842,
            0.843,
            0.905,
            0.928,
            0.939,
            0.947,
            0.951,
            0.954,
            0.957,
            0.958,
            0.959,
            0.960,
            0.961,
        ],  # 0.744
        [
            0.831,
            0.832,
            0.899,
            0.923,
            0.936,
            0.944,
            0.949,
            0.952,
            0.955,
            0.957,
            0.958,
            0.959,
            0.959,
        ],  # 0.808
        [
            0.820,
            0.821,
            0.892,
            0.918,
            0.932,
            0.940,
            0.946,
            0.950,
            0.953,
            0.955,
            0.956,
            0.957,
            0.958,
        ],  # 0.872
        [
            0.807,
            0.808,
            0.884,
            0.912,
            0.927,
            0.936,
            0.942,
            0.947,
            0.950,
            0.952,
            0.954,
            0.955,
            0.956,
        ],  # 0.936
        [
            0.795,
            0.796,
            0.877,
            0.907,
            0.923,
            0.933,
            0.939,
            0.944,
            0.948,
            0.950,
            0.952,
            0.953,
            0.954,
        ],  # 1.000
    ]
).T


class MotorMap(om.Group):
    '''
    This function takes in 0-1 values for electric motor throttle,
    scales those values into 0 to max_torque on the motor map
    this also allows us to solve for motor efficiency
    then we scale the torque up based on the actual scale factor of the motor.
    This avoids the need to rescale the map values, and still allows for the motor scale to be optimized.
    Scaling only effects torque (and therefore shaft power production, and electric power consumption).
    RPM is not scaled and is assumed to be maxed at 6,000 rpm.
    The original maps were put together for a 746kw (1,000 hp) electric motor published in the TTBW paper:
    https://ntrs.nasa.gov/api/citations/20230016987/downloads/TTBW_SciTech_2024_Final_12_5_2023.pdf
    The map is shown in Figure 4.

    Inputs
    ----------
    Dynamic.Mission.THROTTLE : float (unitless) (0 to 1)
        The throttle command which will be translated into torque output from the engine
    Aircraft.Engine.SCALE_FACTOR : float (unitless) (positive)
    Aircraft.Motor.RPM : float (rpm) (0 to 6000)

    Outputs
    ----------
    Dynamic.Mission.TORQUE : float (positive)
    Dynamic.Mission.Motor.EFFICIENCY : float (positive)

    '''

    def initialize(self):
        self.options.declare("num_nodes", types=int)

    def setup(self):
        n = self.options["num_nodes"]

        # Training data
        rpm_vals = (
            np.array(
                [
                    0,
                    0.083333,
                    0.16667,
                    0.25,
                    0.33333,
                    0.41667,
                    0.5,
                    0.58333,
                    0.66667,
                    0.75,
                    0.83333,
                    0.91667,
                    1.0,
                ]
            )
            * 6000
        )
        torque_vals = (
            np.array(
                [
                    0.0,
                    0.040,
                    0.104,
                    0.168,
                    0.232,
                    0.296,
                    0.360,
                    0.424,
                    0.488,
                    0.552,
                    0.616,
                    0.680,
                    0.744,
                    0.808,
                    0.872,
                    0.936,
                    1.000,
                ]
            )
            * 1800
        )

        # Create a structured metamodel to compute motor efficiency from rpm
        motor = om.MetaModelStructuredComp(
            method="slinear", vec_size=n, extrapolate=True
        )
        motor.add_input(
            Dynamic.Mission.RPM, val=np.ones(n), training_data=rpm_vals, units="rpm"
        )
        motor.add_input(
            "torque_unscaled",
            val=np.ones(n),  # unscaled torque
            training_data=torque_vals,
            units="N*m",
        )
        motor.add_output(
            "motor_efficiency",
            val=np.ones(n),
            training_data=motor_map,
            units='unitless',
        )

        self.add_subsystem(
            'throttle_to_torque',
            om.ExecComp(
                'torque_unscaled = torque_max * throttle',
                torque_unscaled={'val': np.ones(n), 'units': 'N*m'},
                torque_max={'val': torque_vals[-1], 'units': 'N*m'},
                throttle={'val': np.ones(n), 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes=[("throttle", Dynamic.Mission.THROTTLE)],
        )

        self.add_subsystem(
            name="motor_efficiency",
            subsys=motor,
            promotes_inputs=[Dynamic.Mission.RPM],
            promotes_outputs=["motor_efficiency"],
        )

        # Now that we know the efficiency, scale up the torque correctly for the engine
        #   size selected
        # Note: This allows the optimizer to optimize the motor size if desired
        self.add_subsystem(
            'scale_motor_torque',
            om.ExecComp(
                'torque = torque_unscaled * scale_factor',
                torque={'val': np.ones(n), 'units': 'N*m'},
                torque_unscaled={'val': np.ones(n), 'units': 'N*m'},
                scale_factor={'val': 1.0, 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes=[
                ("torque", Dynamic.Mission.TORQUE),
                ("scale_factor", Aircraft.Engine.SCALE_FACTOR),
            ],
        )

        self.connect(
            'throttle_to_torque.torque_unscaled',
            ['motor_efficiency.torque_unscaled', 'scale_motor_torque.torque_unscaled'],
        )
