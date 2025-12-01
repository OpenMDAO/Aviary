import numpy as np
import openmdao.api as om

from aviary import constants
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class UnsteadySolvedFlightConditions(om.ExplicitComponent):
    """
    Cross-compute TAS, EAS, and Mach regardless of the input speed type.

    Inputs:
        Dynamic.Atmosphere.DENSITY : local atmospheric density
        Dynamic.Mission.SPEED_OF_SOUND : local speed of sound

    Additional inputs if ground_roll = False:
        Dynamic.Mission.FLIGHT_PATH_ANGLE : flight path angle

    Additional inputs when input_speed_type = SpeedType.TAS:
        Dynamic.Mission.VELOCITY : true airspeed
        dTAS_dr : approximate rate of change of true airspeed per unit range

    Additional inputs when input_speed_type = SpeedType.EAS:
        EAS : equivalent airspeed
        dEAS_dr : approximate rate of change of equivalent airspeed per unit range

    Additional inputs when input_speed_type = SpeedType.MACH:
        Dynamic.Atmosphere.MACH : Mach number
        dmach_dr : approximate rate of change of Mach number per unit range

    Outputs always provided:
        Dynamic.Atmosphere.DYNAMIC_PRESSURE : dynamic pressure
        dTAS_dt_approx : approximate time derivative of TAS based on control rates.

    Additional outputs when input_speed_type = SpeedType.TAS
        EAS : equivalent airspeed
        Dynamic.Atmosphere.MACH : Mach number

    Outputs provided when input_speed_type = SpeedType.EAS:
        TAS : true airspeed
        Dynamic.Atmosphere.MACH : Mach number
    """

    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare(
            'input_speed_type',
            default=SpeedType.TAS,
            types=SpeedType,
            desc='tells whether the input airspeed is equivalent airspeed, true airspeed, or Mach number',
        )
        self.options.declare(
            'ground_roll',
            types=bool,
            default=False,
            desc='True if the aircraft is confined to the ground. '
            'Removes altitude rate as an output and adjust '
            'the TAS rate equation.',
        )

    def setup(self):
        nn = self.options['num_nodes']
        in_type = self.options['input_speed_type']
        ground_roll = self.options['ground_roll']
        ar = np.arange(self.options['num_nodes'])

        self.add_input(
            Dynamic.Atmosphere.DENSITY,
            val=np.zeros(nn),
            units='kg/m**3',
            desc='density of air',
        )
        self.add_input(
            Dynamic.Atmosphere.SPEED_OF_SOUND,
            val=np.zeros(nn),
            units='m/s',
            desc='speed of sound',
        )

        self.add_output(
            Dynamic.Atmosphere.DYNAMIC_PRESSURE,
            val=np.zeros(nn),
            units='N/m**2',
            desc='dynamic pressure',
        )

        self.add_output(
            'dTAS_dt_approx',
            val=np.zeros(nn),
            units='m/s**2',
            desc='approximated rate of change of true airspeed',
        )

        if not ground_roll:
            self.add_input(
                Dynamic.Mission.FLIGHT_PATH_ANGLE,
                shape=nn,
                units='rad',
                desc='flight path angle',
            )

        if in_type is SpeedType.TAS:
            self.add_input(
                Dynamic.Mission.VELOCITY,
                val=np.zeros(nn),
                units='m/s',
                desc='true air speed',
            )

            self.add_input(
                'dTAS_dr',
                val=np.zeros(nn),
                units='m/s/distance_units',
                desc='change in true air speed per unit range',
            )

            self.add_output(
                'EAS',
                val=np.zeros(nn),
                units='m/s',
                desc='equivalent air speed',
            )
            self.add_output(
                Dynamic.Atmosphere.MACH,
                val=np.zeros(nn),
                units='unitless',
                desc='Mach number',
            )

            self.declare_partials(
                of=Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                wrt=[Dynamic.Atmosphere.DENSITY, Dynamic.Mission.VELOCITY],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                of=Dynamic.Atmosphere.MACH,
                wrt=[Dynamic.Atmosphere.SPEED_OF_SOUND, Dynamic.Mission.VELOCITY],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                of='EAS',
                wrt=[Dynamic.Mission.VELOCITY, Dynamic.Atmosphere.DENSITY],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(of='dTAS_dt_approx', wrt=['dTAS_dr'], rows=ar, cols=ar)
            self.declare_partials(
                of='dTAS_dt_approx', wrt=[Dynamic.Mission.VELOCITY], rows=ar, cols=ar
            )

            if not ground_roll:
                self.declare_partials(
                    of='dTAS_dt_approx',
                    wrt=[Dynamic.Mission.FLIGHT_PATH_ANGLE],
                    rows=ar,
                    cols=ar,
                )

        elif in_type is SpeedType.EAS:
            self.add_input(
                'EAS',
                val=np.zeros(nn),
                units='m/s',
                desc='equivalent air speed',
            )
            self.add_input(
                'dEAS_dr',
                val=np.zeros(nn),
                units='1/s',
                desc='change in equivalent air speed per unit range',
            )
            self.add_input(
                'drho_dh',
                val=np.zeros(nn),
                units='kg/m**4',
                desc='change in air density per unit altitude',
            )

            self.add_output(
                Dynamic.Mission.VELOCITY,
                val=np.zeros(nn),
                units='m/s',
                desc='true air speed',
            )
            self.add_output(
                Dynamic.Atmosphere.MACH,
                val=np.zeros(nn),
                units='unitless',
                desc='Mach number',
            )

            self.declare_partials(
                of=Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                wrt=[Dynamic.Atmosphere.DENSITY, 'EAS'],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                of=Dynamic.Atmosphere.MACH,
                wrt=[
                    Dynamic.Atmosphere.SPEED_OF_SOUND,
                    'EAS',
                    Dynamic.Atmosphere.DENSITY,
                ],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                of=Dynamic.Mission.VELOCITY,
                wrt=[Dynamic.Atmosphere.DENSITY, 'EAS'],
                rows=ar,
                cols=ar,
            )
            self.declare_partials(
                of='dTAS_dt_approx',
                wrt=['drho_dh', Dynamic.Atmosphere.DENSITY, 'EAS', 'dEAS_dr'],
                rows=ar,
                cols=ar,
            )

            if not ground_roll:
                self.declare_partials(
                    of='dTAS_dt_approx',
                    wrt=[Dynamic.Mission.FLIGHT_PATH_ANGLE],
                    rows=ar,
                    cols=ar,
                )

        else:
            self.add_input(
                Dynamic.Atmosphere.MACH,
                val=np.zeros(nn),
                units='unitless',
                desc='Mach number',
            )
            self.add_input(
                'dmach_dr',
                val=np.zeros(nn),
                units='1/distance_units',
                desc='change in Mach number per unit range',
            )
            self.add_input(
                'dsos_dh',
                val=np.zeros(nn),
                units='m/s/m',
                desc='change in speed of sound per unit altitude',
            )

            self.add_output(
                'EAS',
                val=np.zeros(nn),
                units='m/s',
                desc='equivalent air speed',
            )
            self.add_output(
                Dynamic.Mission.VELOCITY,
                val=np.zeros(nn),
                units='m/s',
                desc='true air speed',
            )

            self.declare_partials(
                of=Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                wrt=[
                    Dynamic.Atmosphere.SPEED_OF_SOUND,
                    Dynamic.Atmosphere.MACH,
                    Dynamic.Atmosphere.DENSITY,
                ],
                rows=ar,
                cols=ar,
            )

            self.declare_partials(
                of=Dynamic.Mission.VELOCITY,
                wrt=[Dynamic.Atmosphere.SPEED_OF_SOUND, Dynamic.Atmosphere.MACH],
                rows=ar,
                cols=ar,
            )

            self.declare_partials(
                of='EAS',
                wrt=[
                    Dynamic.Atmosphere.SPEED_OF_SOUND,
                    Dynamic.Atmosphere.MACH,
                    Dynamic.Atmosphere.DENSITY,
                ],
                rows=ar,
                cols=ar,
            )

            self.declare_partials(
                of='dTAS_dt_approx', wrt=['dmach_dr', 'dsos_dh'], rows=ar, cols=ar
            )

    def compute(self, inputs, outputs):
        in_type = self.options['input_speed_type']
        ground_roll = self.options['ground_roll']

        rho = inputs[Dynamic.Atmosphere.DENSITY]
        rho_sl = constants.RHO_SEA_LEVEL_METRIC
        sqrt_rho_rho_sl = np.sqrt(rho / rho_sl)
        sos = inputs[Dynamic.Atmosphere.SPEED_OF_SOUND]

        cgam = 1.0 if ground_roll else np.cos(inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE])
        sgam = 0.0 if ground_roll else np.sin(inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE])

        if in_type is SpeedType.TAS:
            tas = inputs[Dynamic.Mission.VELOCITY]
            dtas_dr = inputs['dTAS_dr']
            outputs[Dynamic.Atmosphere.MACH] = tas / sos
            outputs['EAS'] = tas * sqrt_rho_rho_sl
            outputs['dTAS_dt_approx'] = dtas_dr * tas * cgam

        elif in_type is SpeedType.EAS:
            eas = inputs['EAS']
            drho_dh = inputs['drho_dh']
            deas_dr = inputs['dEAS_dr']
            outputs[Dynamic.Mission.VELOCITY] = tas = eas / sqrt_rho_rho_sl
            outputs[Dynamic.Atmosphere.MACH] = tas / sos
            drho_dt_approx = drho_dh * tas * sgam
            deas_dt_approx = deas_dr * tas * cgam
            outputs['dTAS_dt_approx'] = (
                deas_dt_approx * (rho_sl / rho) ** 1.5
                - 0.5 * eas * drho_dt_approx * rho_sl**1.5 / rho_sl**2.5
            )

        else:
            mach = inputs[Dynamic.Atmosphere.MACH]
            dmach_dr = inputs['dmach_dr']
            outputs[Dynamic.Mission.VELOCITY] = tas = sos * mach
            outputs['EAS'] = tas * sqrt_rho_rho_sl
            dmach_dt_approx = dmach_dr * tas * cgam
            dsos_dt_approx = inputs['dsos_dh'] * tas * sgam
            outputs['dTAS_dt_approx'] = dmach_dt_approx * sos + dsos_dt_approx * tas / sos

        outputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE] = 0.5 * rho * tas**2

    def compute_partials(self, inputs, partials):
        in_type = self.options['input_speed_type']
        ground_roll = self.options['ground_roll']

        rho = inputs[Dynamic.Atmosphere.DENSITY]
        rho_sl = constants.RHO_SEA_LEVEL_METRIC
        sqrt_rho_rho_sl = np.sqrt(rho / rho_sl)
        dsqrt_rho_rho_sl_drho = 0.5 / sqrt_rho_rho_sl / rho_sl
        sos = inputs[Dynamic.Atmosphere.SPEED_OF_SOUND]

        cgam = 1.0 if ground_roll else np.cos(inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE])
        sgam = 0.0 if ground_roll else np.sin(inputs[Dynamic.Mission.FLIGHT_PATH_ANGLE])

        if in_type is SpeedType.TAS:
            TAS = inputs[Dynamic.Mission.VELOCITY]  # Why is there tas and TAS?

            tas = inputs[Dynamic.Mission.VELOCITY]
            dTAS_dr = inputs['dTAS_dr']

            partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Mission.VELOCITY] = rho * TAS
            partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.DENSITY] = 0.5 * TAS**2

            partials[Dynamic.Atmosphere.MACH, Dynamic.Mission.VELOCITY] = 1 / sos
            partials[Dynamic.Atmosphere.MACH, Dynamic.Atmosphere.SPEED_OF_SOUND] = -TAS / sos**2

            partials['EAS', Dynamic.Mission.VELOCITY] = sqrt_rho_rho_sl
            partials['EAS', Dynamic.Atmosphere.DENSITY] = tas * dsqrt_rho_rho_sl_drho

            partials['dTAS_dt_approx', 'dTAS_dr'] = tas * cgam
            partials['dTAS_dt_approx', Dynamic.Mission.VELOCITY] = dTAS_dr * cgam

            if not ground_roll:
                partials['dTAS_dt_approx', Dynamic.Mission.FLIGHT_PATH_ANGLE] = (
                    -dTAS_dr * tas * sgam
                )

        elif in_type is SpeedType.EAS:
            EAS = inputs['EAS']
            TAS = EAS / sqrt_rho_rho_sl

            dTAS_dRho = -0.5 * EAS * rho_sl**0.5 / rho**1.5
            dTAS_dEAS = 1 / sqrt_rho_rho_sl

            partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, 'EAS'] = EAS * rho_sl
            partials[Dynamic.Atmosphere.MACH, 'EAS'] = dTAS_dEAS / sos
            partials[Dynamic.Atmosphere.MACH, Dynamic.Atmosphere.DENSITY] = dTAS_dRho / sos
            partials[Dynamic.Atmosphere.MACH, Dynamic.Atmosphere.SPEED_OF_SOUND] = -TAS / sos**2
            partials[Dynamic.Mission.VELOCITY, Dynamic.Atmosphere.DENSITY] = dTAS_dRho
            partials[Dynamic.Mission.VELOCITY, 'EAS'] = dTAS_dEAS
            partials['dTAS_dt_approx', 'dEAS_dr'] = TAS * cgam * (rho_sl / rho) ** 1.5
            partials['dTAS_dt_approx', 'drho_dh'] = (
                -0.5 * EAS * TAS * sgam * rho_sl**1.5 / rho_sl**2.5
            )

        else:
            mach = inputs[Dynamic.Atmosphere.MACH]
            TAS = sos * mach

            partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.SPEED_OF_SOUND] = (
                rho * sos * mach**2
            )
            partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.MACH] = (
                rho * sos**2 * mach
            )
            partials[Dynamic.Atmosphere.DYNAMIC_PRESSURE, Dynamic.Atmosphere.DENSITY] = (
                0.5 * sos**2 * mach**2
            )
            partials[Dynamic.Mission.VELOCITY, Dynamic.Atmosphere.SPEED_OF_SOUND] = mach
            partials[Dynamic.Mission.VELOCITY, Dynamic.Atmosphere.MACH] = sos
            partials['EAS', Dynamic.Atmosphere.SPEED_OF_SOUND] = mach * sqrt_rho_rho_sl
            partials['EAS', Dynamic.Atmosphere.MACH] = sos * sqrt_rho_rho_sl
            partials['EAS', Dynamic.Atmosphere.DENSITY] = (
                TAS * (1 / rho_sl) ** 0.5 * 0.5 * rho ** (-0.5)
            )
            partials['dTAS_dt_approx', 'dmach_dr'] = TAS * cgam * sos
            partials['dTAS_dt_approx', 'dsos_dh'] = TAS**2 * sgam / sos
