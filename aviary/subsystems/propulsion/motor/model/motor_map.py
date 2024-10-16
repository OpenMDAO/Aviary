import numpy as np

import openmdao.api as om

from aviary.variable_info.variables import Dynamic, Aircraft


motor_map = np.array([
    # speed---- .0       .083333  .16667  .25    .33333.41667  .5,     .58333 .66667  .75,   .83333, .91667 1.
    [.871,    .872,    .862,   .853,  .845, .838,   .832,   .825,  .819,   .813,  .807,   .802,   .796],  # 0
    [.872,    .873,    .863,   .854,  .846, .839,   .833,   .826,  .820,   .814,   .808,  .803,   .797],  # 0.040
    [.928,    .928,    .932,   .930,  .928, .926,   .923,   .920,  .918,   .915,   .912,  .909,   .907],  # 0.104
    [.931,    .932,    .944,   .947,  .947, .947,   .946,   .945,  .943,   .942,   .940,  .938,   .937],  # 0.168
    [.931,    .927,    .946,   .952,  .954, .955,   .955,   .954,  .954,   .953,   .952,  .951,   .950],  # 0.232
    [.917,    .918,    .944,   .952,  .956, .958,   .958,   .959,  .959,   .958,   .958,  .957,   .956],  # 0.296
    [.907,    .908,    .940,   .951,  .956, .958,   .960,   .961,  .961,   .961,   .961,  .960,   .960],  # 0.360
    [.897,    .898,    .935,   .948,  .954, .958,   .960,   .961,  .962,   .962,   .962,  .962,   .962],  # 0.424
    [.886,    .887,    .930,   .945,  .952, .956,   .959,   .960,  .962,   .962,   .963,  .963,   .963],  # 0.488
    [.875,    .876,    .924,   .941,  .949, .954,   .957,   .960,  .961,   .962,   .962,  .963,   .963],  # 0.552
    [.864,    .865,    .918,   .937,  .946, .952,   .956,   .958,  .960,   .961,   .962,  .962,   .962],  # 0.616
    [.853,    .854,    .912,   .932,  .943, .949,   .953,   .956,  .958,   .960,   .961,  .961,   .962],  # 0.680
    [.842,    .843,    .905,   .928,  .939, .947,   .951,   .954,  .957,   .958,   .959,  .960,   .961],  # 0.744
    [.831,    .832,    .899,   .923,  .936, .944,   .949,   .952,  .955,   .957,   .958,  .959,   .959],  # 0.808
    [.820,    .821,    .892,   .918,  .932, .940,   .946,   .950,  .953,   .955,   .956,  .957,   .958],  # 0.872
    [.807,    .808,    .884,   .912,  .927, .936,   .942,   .947,  .950,   .952,   .954,  .955,   .956],  # 0.936
    [.795,    .796,    .877,   .907,  .923, .933,   .939,   .944,  .948,   .950,   .952,  .953,   .954]  # 1.000
]).T


class MotorMap(om.Group):

    '''
    This function takes in 0-1 values for electric motor throttle,
    scales those values into 0 to max_torque on the motor map
    this also allows us to solve for motor efficiency
    then we scale the torque up based on the actual scale factor of the motor.
    This avoids the need to rescale the map values, and still allows for the motor scale to be optimized.
    Scaling only effects Torque. RPM is not scaled and is assumed to be maxed at 6,000 rpm.
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
        rpm_vals = np.array([0, .083333, .16667, .25, .33333, .41667, .5,
                            .58333, .66667, .75, .83333, .91667, 1.])*6000
        torque_vals = np.array([0.0, 0.040, 0.104, 0.168, 0.232, 0.296, 0.360,
                                0.424, 0.488, 0.552, 0.616, 0.680, 0.744, 0.808,
                                0.872, 0.936, 1.000])*1800

        # Create a structured metamodel to compute motor efficiency from rpm
        motor = om.MetaModelStructuredComp(method="slinear",
                                           vec_size=n,
                                           extrapolate=True)
        motor.add_input(Dynamic.Mission.RPM, val=np.ones(n),
                        training_data=rpm_vals,
                        units="rpm")
        motor.add_input("torque_unscaled", val=np.ones(n),  # unscaled torque
                        training_data=torque_vals,
                        units="N*m")
        motor.add_output("motor_efficiency", val=np.ones(n),
                         training_data=motor_map,
                         units='unitless')

        self.add_subsystem(
            'throttle_to_torque',
            om.ExecComp(
                'torque_unscaled = torque_max * throttle',
                torque_unscaled={'val': np.ones(n), 'units': 'N*m'},
                torque_max={'val': torque_vals[-1], 'units': 'N*m'},
                throttle={'val': np.ones(n), 'units': 'unitless'},
                has_diag_partials=True,
            ),
            promotes=["torque_unscaled", ("throttle", Dynamic.Mission.THROTTLE)],
        )

        self.add_subsystem(name="motor_efficiency",
                           subsys=motor,
                           promotes_inputs=[Dynamic.Mission.RPM, "torque_unscaled"],
                           promotes_outputs=["motor_efficiency"])

        # now that we know the efficiency, scale up the torque correctly for the engine size selected
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
                "torque_unscaled",
                ("scale_factor", Aircraft.Engine.SCALE_FACTOR),
            ],
        )
