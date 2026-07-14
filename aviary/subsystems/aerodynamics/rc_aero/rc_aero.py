import numpy as np
import openmdao.api as om

from ambiance import Atmosphere

from aviary.variable_info.functions import add_aviary_input
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.subsystems.aerodynamics.rc_aero.OAS_aero_analysis import OASAero

class FuselageDrag(om.ExplicitComponent):
    # based on Roskam VI chapter 4
    # assuming rectangular cross section rather than elliptical (hence max height and width rather than diameter)
    # assuming fuselage Reynolds number of about 1mil at a very low Mach number

    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # constants for a DBF-sized RC aircraft
        add_aviary_input(self, Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR, 'unitless') # say 1.2 from figure 4.1
        self.add_input('Cf_fus', 0.0044, units='unitless') # turbulent flat plate skin friction coeff, from figure 4.3
        self.add_input('CD_L_fus', 0.001, units='unitless') # assume little lift induced drag from fuselage

        add_aviary_input(self, Aircraft.Fuselage.LENGTH, units='m')
        add_aviary_input(self, Aircraft.Fuselage.MAX_HEIGHT, units='m') 
        add_aviary_input(self, Aircraft.Fuselage.MAX_WIDTH, units='m')
        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')

        self.add_output(name='D_fus', shape=(nn,), units='N') # fuselage drag
        self.add_output(name='CD_fus', shape=(nn,), units='unitless') # fuselage CD0

        self.declare_partials('D_fus', [Aircraft.Fuselage.LENGTH,
                                        Aircraft.Fuselage.MAX_HEIGHT,
                                        Aircraft.Fuselage.MAX_WIDTH,
                                        Aircraft.Wing.AREA],
                                        method='cs')
        
    def compute(self, inputs, outputs): 
        R_wf = inputs[Aircraft.Wing.FUSELAGE_INTERFERENCE_FACTOR]
        Cf_fus = inputs['Cf_fus']
        CD_L_fus = inputs['CD_L_fus']

        lf = inputs[Aircraft.Fuselage.LENGTH]
        df = inputs[Aircraft.Fuselage.MAX_HEIGHT]
        wf = inputs[Aircraft.Fuselage.MAX_WIDTH]
        S_ref = inputs[Aircraft.Wing.AREA]
        db = df / 3 # base diameter

        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE] # from aero conditions component

        S_fus = wf * df # fuselage maximum frontal area
        S_wet = 2 * ((lf * df) + (lf * wf)) # very basic stuff. just the sides

        CD0_fus_base = R_wf * Cf_fus * (1 + 60 / ((lf/df)**3) + 
                                        0.0025 * (lf/df)) * S_wet / S_ref # zero lift drag coeff exculsive of base
        CDb_fus = ((0.029 * ((db/df)**3) / 
                    (CD0_fus_base * np.sqrt(S_ref/S_fus)))) * (S_fus/S_ref) # fuselage base drag coeff

        CD0_fus = CD0_fus_base + CDb_fus # total fuselage zero-lift drag coeff

        CD_fus = CD0_fus +CD_L_fus # total fuselage drag coeff

        outputs['D_fus'] = q * S_ref * CD_fus
        outputs['CD_fus'] = CD_fus

class VTailDrag(om.ExplicitComponent):
    # based on Roskam VI chapter 4
    # assuming fuselage Reynolds number of about 1mil at a very low Mach number
    
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        # constants for a DBF-sized RC aircraft
        self.add_input('R_LS', 1.1, units='unitless') # lifting surface correction factor wt low sweep @ low Mach, from fig 4.2
        self.add_input('Cf_vtail', 0.0044, units='unitless') # turbulent flat plate skin friction coeff, from figure 4.3
        self.add_input('L_prime', 1.2, units='unitless') # airfoil thickness location parameter, for (t/c)max @ x_t >= 0.3c (NACA 00 series)

        add_aviary_input(self, Aircraft.VerticalTail.ROOT_CHORD)
        add_aviary_input(self, Aircraft.VerticalTail.TAPER_RATIO)
        add_aviary_input(self, Aircraft.VerticalTail.SPAN)
        add_aviary_input(self, Aircraft.Wing.AREA)
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn)
        add_aviary_input(self, Aircraft.VerticalTail.THICKNESS_TO_CHORD)
        
        self.add_output(name='D_vtail', shape=(nn,), units='N') # vtail drag
        self.add_output(name='CD_vtail', shape=(nn,), units='unitless') # vtail CD0

        self.declare_partials('D_vtail', [Aircraft.VerticalTail.ROOT_CHORD,
                                          Aircraft.VerticalTail.TAPER_RATIO,
                                          Aircraft.VerticalTail.SPAN,
                                          Aircraft.Wing.AREA])
        
    def compute(self, inputs, outputs):
        R_LS = inputs['R_LS']
        Cf_vtail = inputs['Cf_vtail']
        L_prime = inputs['L_prime']

        c = inputs[Aircraft.VerticalTail.ROOT_CHORD]
        taper = inputs[Aircraft.VerticalTail.TAPER_RATIO]
        b = inputs[Aircraft.VerticalTail.SPAN]
        S_ref_wing = inputs[Aircraft.Wing.AREA]

        t_on_c = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]

        S_ref_vtail = ((c + (taper * c)) * b) / 2 # planform area
        S_wet_vtail = S_ref_vtail * 2 # add area of airfoil to this (end of wing)
        CD0_vtail = R_LS * Cf_vtail * (1 + (L_prime * t_on_c) + 100 * ((t_on_c)**4)) * S_wet_vtail / S_ref_wing

        outputs['D_vtail'] = q * S_ref_vtail * CD0_vtail
        outputs['CD_vtail'] = CD0_vtail # lift induced negligible

    def compute_partials(self, inputs, partials):
        R_LS = inputs['R_LS']
        Cf_vtail = inputs['Cf_vtail']
        L_prime = inputs['L_prime']

        c = inputs[Aircraft.VerticalTail.ROOT_CHORD]
        taper = inputs[Aircraft.VerticalTail.TAPER_RATIO]
        b = inputs[Aircraft.VerticalTail.SPAN]
        S_ref_wing = inputs[Aircraft.Wing.AREA]

        t_on_c = inputs[Aircraft.VerticalTail.THICKNESS_TO_CHORD]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]

        partials['D_vtail', Aircraft.VerticalTail.ROOT_CHORD] = b**2 * Cf_vtail * q * R_LS * c * (taper + 1)**2 * ((L_prime * t_on_c)* + 100 * ((t_on_c)**4) + 1) / S_ref_wing
        partials['D_vtail', Aircraft.VerticalTail.TAPER_RATIO] = b**2 * c**2 * Cf_vtail * q * R_LS * c * (taper + 1) * ((L_prime * t_on_c)* + 100 * ((t_on_c)**4) + 1) / S_ref_wing
        partials['D_vtail', Aircraft.VerticalTail.SPAN] = Cf_vtail * q * R_LS * b * ((c * taper) + c)**2 * ((L_prime * t_on_c)* + 100 * ((t_on_c)**4) + 1) / S_ref_wing
        partials['D_vtail', Aircraft.Wing.AREA] = -0.5 * b**2 * Cf_vtail * q * R_LS * ((c * taper) + c)**2 * ((L_prime * t_on_c)* + 100 * ((t_on_c)**4) + 1) / S_ref_wing**2

class LandingGearDrag(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.LandingGear.DRAG_COEFFICIENT, units='unitless')
        add_aviary_input(self, Dynamic.Atmosphere.DYNAMIC_PRESSURE, shape=nn, units='N/m**2')
        add_aviary_input(self, Aircraft.Wing.AREA, units='m**2')

        self.add_output('D_gear', shape=(nn,), units='N')
        self.add_output('CD_gear', shape=(nn,), units='unitless')

        self.declare_partials('D_gear', [Aircraft.LandingGear.DRAG_COEFFICIENT,
                                         Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                                         Aircraft.Wing.AREA])

    def compute(self, inputs, outputs):
        CD_gear = inputs[Aircraft.LandingGear.DRAG_COEFFICIENT]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        S_ref = inputs[Aircraft.Wing.AREA]

        outputs['D_gear'] = CD_gear * q * S_ref
        outputs['CD_gear'] = CD_gear

    def compute_partials(self, inputs, partials):
        CD_gear = inputs[Aircraft.LandingGear.DRAG_COEFFICIENT]
        q = inputs[Dynamic.Atmosphere.DYNAMIC_PRESSURE]
        S_ref = inputs[Aircraft.Wing.AREA]

        nn = self.options['num_nodes']
        partials['D_gear', Aircraft.LandingGear.DRAG_COEFFICIENT] = q * S_ref
        partials['D_gear', Dynamic.Atmosphere.DYNAMIC_PRESSURE] = CD_gear * S_ref
        partials['D_gear', Aircraft.Wing.AREA] = CD_gear * q

class Averages(om.ExplicitComponent):
    # averages because Aviary objectives must be scalar. not doing this would be preferable
    def initialize(self):
        self.options.declare('num_nodes', types=int)

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('CD', shape=nn, units='unitless')
        self.add_output('avg_CD', units='unitless')

        self.add_input('CD_fus', shape=nn, units='unitless')
        self.add_output('avg_CD_fus', units='unitless')

        self.add_input('lifting_surface_CL', shape=nn, units='unitless')
        self.add_output('avg_CL', units='unitless')

        self.declare_partials('avg_CD', 'CD')
        self.declare_partials('avg_CD_fus', 'CD_fus')
        self.declare_partials('avg_CL', 'lifting_surface_CL')

    def compute(self, inputs, outputs):
        total_CD = inputs['CD']
        outputs['avg_CD'] = np.mean(total_CD)

        total_CD_fus = inputs['CD_fus']
        outputs['avg_CD_fus'] = np.mean(total_CD_fus)

        total_CL = inputs['lifting_surface_CL']
        outputs['avg_CL'] = np.mean(total_CL)

    def compute_partials(self, inputs, partials):
        nn = self.options['num_nodes']
        partials['avg_CD', 'CD'] = np.ones(nn) / nn
        partials['avg_CD_fus', 'CD_fus'] = np.ones(nn) / nn
        partials['avg_CL', 'lifting_surface_CL'] = np.ones(nn) / nn


class TotalAircraftAero(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', types=int)
        self.options.declare('aviary_inputs')

    def setup(self):
        nn = self.options['num_nodes']
        aviary_inputs = self.options['aviary_inputs']

        self.add_subsystem(
            'OAS_aero',
            OASAero(num_nodes=nn, aviary_inputs=aviary_inputs),
            promotes_inputs=['*'],
            promotes_outputs=[
                'lifting_surface_drag', 
                Dynamic.Vehicle.LIFT, 
                Dynamic.Atmosphere.DYNAMIC_PRESSURE,
                'lifting_surface_CL',
                'lifting_surface_CD',
                'alpha'
                ]
        )

        self.add_subsystem(
            'fuselage_drag',
            FuselageDrag(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['CD_fus', 'D_fus']
        )

        self.add_subsystem(
            'vtail_drag',
            VTailDrag(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['CD_vtail', 'D_vtail']
        )

        self.add_subsystem(
            'landing_gear_drag',
            LandingGearDrag(num_nodes=nn),
            promotes_inputs=['*'],
            promotes_outputs=['CD_gear', 'D_gear']
        )

        self.add_subsystem(
            'total_aircraft_CD',
            om.ExecComp('CD = CD_fus + CD_vtail + lifting_surface_CD + CD_gear',
                        CD_fus={'shape': (nn,), 'units': 'unitless'},
                        CD_vtail={'shape': (nn,), 'units': 'unitless'},
                        lifting_surface_CD={'shape': (nn,), 'units': 'unitless'},
                        CD_gear={'shape': (nn,), 'units': 'unitless'},
                        CD={'shape': (nn,), 'units': 'unitless'}),
            promotes_inputs=['CD_fus', 'CD_vtail', 'lifting_surface_CD', 'CD_gear'],
            promotes_outputs=['CD']
        )

        self.add_subsystem(
            'total_aircraft_drag',
            om.ExecComp('drag = D_fus + D_vtail + lifting_surface_drag + D_gear',
                        D_fus={'shape': (nn,), 'units': 'N'},
                        D_vtail={'shape': (nn,), 'units': 'N'},
                        lifting_surface_drag={'shape': (nn,), 'units': 'N'},
                        D_gear={'shape': (nn,), 'units': 'N'},
                        drag={'shape': (nn,), 'units': 'N'}),
            promotes_inputs=['D_fus', 'D_vtail', 'lifting_surface_drag', 'D_gear'],
            promotes_outputs=[Dynamic.Vehicle.DRAG]
        )

        # would like to not need this
        self.add_subsystem(
            'averages',
            Averages(num_nodes=nn),
            promotes_inputs=['CD', 'CD_fus', 'lifting_surface_CL'],
            promotes_outputs=['avg_CD', 'avg_CD_fus', 'avg_CL']
        )
        
        self.connect('OAS_aero.aero_point_0.wing.S_ref', 'aircraft:wing:area')
        self.connect('aircraft:wing:root_chord', 'OAS_aero.aero_point_0.wing.c_root')
        
        self.options['auto_order'] = True