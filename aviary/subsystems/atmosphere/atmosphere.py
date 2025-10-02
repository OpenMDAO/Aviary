import openmdao.api as om
from aviary.subsystems.atmosphere.atmos_1976 import USatm1976Comp

from aviary.subsystems.atmosphere.flight_conditions import FlightConditions
from aviary.variable_info.enums import SpeedType
from aviary.variable_info.variables import Dynamic


class Atmosphere(om.Group):
    """
    Group that contains atmospheric conditions for the aircraft's current flight
    condition, as well as conversions for different speed types (TAS, EAS, Mach).
    """

    def initialize(self):
        self.options.declare(
            'num_nodes', types=int, desc='Number of nodes to be evaluated in the RHS'
        )

        self.options.declare(
            'h_def',
            values=('geopotential', 'geodetic'),
            default='geodetic',
            desc='The definition of altitude provided as input to the component. If '
            '"geodetic", it will be converted to geopotential based on Equation 19 in '
            'the original standard.',
        )

        self.options.declare(
            'output_dsos_dh',
            types=bool,
            default=False,
            desc='If true, the derivative of the speed of sound will be added as an output',
        )

        self.options.declare(
            'input_speed_type',
            default=SpeedType.TAS,
            types=SpeedType,
            desc='defines input airspeed as equivalent airspeed, true airspeed, or mach number',
        )

        self.options.declare(
            'output_abs_humidity', 
            types=bool, 
            default=True,
            desc='If true, absolute humidity derived from an empirical model will be added as an output'
        )

        self.options.declare(
            'rel_humidity_sl', 
            types=float, 
            default=0.70,
            desc='Fraction relative humidity at sea level to be used in absolute humidity model'
        )

        self.options.declare(
            'isa_delta_T',
            default=0.0,
            desc='Temperature delta from International Standard Atmosphere (ISA) standard day conditions (degrees Rankine)',
        )

    def setup(self):
        nn = self.options['num_nodes']
        speed_type = self.options['input_speed_type']
        h_def = self.options['h_def']
        output_dsos_dh = self.options['output_dsos_dh']
        output_abs_humidity = self.options['output_abs_humidity']
        rel_humidity_sl = self.options['rel_humidity_sl']
        isa_delta_T = self.options['isa_delta_T']

        self.add_subsystem(
            name='standard_atmosphere',
            subsys=USatm1976Comp(
                num_nodes=nn, 
                h_def=h_def, 
                output_dsos_dh=output_dsos_dh, 
                output_abs_humidity=output_abs_humidity, 
                rel_humidity_sl=rel_humidity_sl,
                isa_delta_T=isa_delta_T),
            promotes_inputs=[('h', Dynamic.Mission.ALTITUDE)],
            promotes_outputs=[
                '*',
                ('sos', Dynamic.Atmosphere.SPEED_OF_SOUND),
                ('rho', Dynamic.Atmosphere.DENSITY),
                ('temp', Dynamic.Atmosphere.TEMPERATURE),
                ('pres', Dynamic.Atmosphere.STATIC_PRESSURE),
            ],
        )

        self.add_subsystem(
            name='flight_conditions',
            subsys=FlightConditions(num_nodes=nn, input_speed_type=speed_type),
            promotes=['*'],
        )
