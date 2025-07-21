import numpy as np

import openmdao.api as om

# DBF mass components
from aviary.examples.external_subsystems.dbf_based_mass.dbf_fuselage import DBFFuselageMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_verticaltail import DBFVerticalTailMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_horizontaltail import DBFHorizontalTailMass
from aviary.examples.external_subsystems.dbf_based_mass.dbf_wing import DBFWingMass
from aviary.examples.external_subsystems.dbf_based_mass.mass_summation import MassSummation
from aviary.variable_info.variables import Aircraft


def run_level3_dbf_example():
    prob = om.Problem()
    model = prob.model

    # -----------------------------
    # Configure fuselage
    # -----------------------------
    
    fuselage = DBFFuselageMass()
    ribs = np.array([0]*14 + [1]*5 + [2])
    bulkhead_materials = np.where(ribs != 0, 'Ply', 'Balsa').tolist()
    rib_thicks = np.where(ribs == 2, 0.25, 0.125)
    fuselage.options['bulkhead_materials'] = bulkhead_materials
    fuselage.set_option(Aircraft.Fuselage.Dbf.BULKHEAD_THICKNESS, val=rib_thicks, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.NUM_SPARS, val=0.5, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.BULKHEAD_LIGHTENING_FACTOR, val=0.18, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_COVERAGE, val=1, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_LIGHTENING_FACTOR, val=0.3, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.GLUE_FACTOR, val=0.08, units='unitless')
    fuselage.set_option(Aircraft.Fuselage.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.FLOOR_LENGTH, val=2, units='ft')
    fuselage.set_option(Aircraft.Fuselage.Dbf.FLOOR_DENSITY, val=340, units='kg/m**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.FLOOR_THICKNESS, val=0.125, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SPAR_DENSITY, val=2, units='g/cm**3')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SPAR_OUTER_DIAMETER, val=1, units='inch')
    fuselage.set_option(Aircraft.Fuselage.Dbf.SPAR_WALL_THICKNESS, val=0.0625, units='inch')
    model.add_subsystem('fuselage', fuselage, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Configure vertical tail
    # -----------------------------
    vtail = DBFVerticalTailMass()
    rib_thicks_v = np.array([0.125]*5)
    rib_materials_v = ['Balsa'] * 4 + ['Ply'] * 1
    vtail.options['rib_materials'] = rib_materials_v
    vtail.options['airfoil_data_file'] = (
        r'aviary\examples\external_subsystems\dbf_based_mass\n0012-il.csv'
    )
    vtail.set_option(Aircraft.VerticalTail.Dbf.RIB_THICKNESS, val=rib_thicks_v, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_COVERAGE, val=0.7, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_LIGHTENING_FACTOR, val=1.0, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    vtail.set_option(Aircraft.VerticalTail.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.NUM_STRINGERS, val=2.5, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.GLUE_FACTOR, val=0.05, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.NUM_SPARS, val=0, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.RIB_LIGHTENING_FACTOR, val=2/3, units='unitless')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SPAR_DENSITY, val=0, units='g/cm**3')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SPAR_OUTER_DIAMETER, val=0, units='inch')
    vtail.set_option(Aircraft.VerticalTail.Dbf.SPAR_WALL_THICKNESS, val=0.0, units='inch')
    model.add_subsystem('vtail', vtail, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Configure horizontal tail
    # -----------------------------
    htail = DBFHorizontalTailMass()
    rib_thicks_h = np.array([0.125]*8)
    rib_materials_h = ['Balsa'] * 6 + ['Ply'] * 2
    htail.options['rib_materials'] = rib_materials_h
    htail.options['airfoil_data_file'] = (
        r'aviary\examples\external_subsystems\dbf_based_mass\n0012-il.csv'
    )
    htail.set_option(Aircraft.HorizontalTail.Dbf.RIB_THICKNESS, val=rib_thicks_h, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_COVERAGE, val=0.7, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_LIGHTENING_FACTOR, val=1.0, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    htail.set_option(Aircraft.HorizontalTail.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.NUM_STRINGERS, val=2.5, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.GLUE_FACTOR, val=0.05, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.NUM_SPARS, val=0, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.RIB_LIGHTENING_FACTOR, val=2/3, units='unitless')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SPAR_DENSITY, val=0, units='g/cm**3')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SPAR_OUTER_DIAMETER, val=0, units='inch')
    htail.set_option(Aircraft.HorizontalTail.Dbf.SPAR_WALL_THICKNESS, val=0, units='inch')
    model.add_subsystem('htail', htail, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Configure wing
    # -----------------------------
    wing = DBFWingMass()
    ribs = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    rib_materials = ['Balsa'] * 15 + ['Ply'] * 5
    rib_thicks = np.where(ribs != 0, 0.125, 0.125)
    wing.options['rib_materials'] = rib_materials
    wing.options['airfoil_data_file'] = (
        r'aviary\examples\external_subsystems\dbf_based_mass\mh84-il.csv'
    )
    wing.set_option(Aircraft.Wing.Dbf.SHEETING_COVERAGE, val=0.4, units='unitless')
    wing.set_option(Aircraft.Wing.Dbf.SHEETING_DENSITY, val=160, units='kg/m**3')
    wing.set_option(Aircraft.Wing.Dbf.SHEETING_LIGHTENING_FACTOR, val=1, units='unitless')
    wing.set_option(Aircraft.Wing.Dbf.SHEETING_THICKNESS, val=0.03125, units='inch')
    wing.set_option(Aircraft.Wing.Dbf.STRINGER_DENSITY, val=160, units='kg/m**3')
    wing.set_option(Aircraft.Wing.Dbf.STRINGER_THICKNESS, val=0.375, units='inch')
    wing.set_option(Aircraft.Wing.Dbf.NUM_STRINGERS, val=2.5, units='unitless')
    wing.set_option(Aircraft.Wing.Dbf.GLUE_FACTOR, val=0.15, units='unitless')
    wing.set_option(Aircraft.Wing.Dbf.NUM_SPARS, val=1.1, units='unitless')
    wing.set_option(Aircraft.Wing.Dbf.RIB_LIGHTENING_FACTOR, val=2/3, units='unitless')
    wing.set_option(Aircraft.Wing.Dbf.RIB_THICKNESS, val=rib_thicks, units='inch')
    wing.set_option(Aircraft.Wing.Dbf.SKIN_DENSITY, val=20, units='g/m**2')
    wing.set_option(Aircraft.Wing.Dbf.SPAR_DENSITY, val=2, units='g/cm**3')
    wing.set_option(Aircraft.Wing.Dbf.SPAR_OUTER_DIAMETER, val=1, units='inch')
    wing.set_option(Aircraft.Wing.Dbf.SPAR_WALL_THICKNESS, val=0.0625, units='inch')
    wing.set_option(Aircraft.Wing.Dbf.MISC_MASS, val=0.0, units='kg')
    model.add_subsystem('wing', wing, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Total mass calculation
    # -----------------------------
    model.add_subsystem('mass_group', MassSummation(), promotes_inputs=['*'], promotes_outputs=['*'])


    # -----------------------------
    # Setup + Run
    # -----------------------------
    prob.setup()

    prob.set_val(Aircraft.Fuselage.LENGTH, 4, units='ft')
    prob.set_val(Aircraft.Fuselage.AVG_HEIGHT, 5, units='inch')
    prob.set_val(Aircraft.Fuselage.AVG_WIDTH, 4, units='inch')
    prob.set_val(Aircraft.Fuselage.WETTED_AREA, 904, units='inch**2')

    prob.set_val(Aircraft.VerticalTail.ROOT_CHORD, 8.75, units='inch')
    prob.set_val(Aircraft.VerticalTail.SPAN, 1, units='ft')
    prob.set_val(Aircraft.VerticalTail.WETTED_AREA, 0.139, units='m**2')

    prob.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 8.75, units='inch')
    prob.set_val(Aircraft.HorizontalTail.SPAN, 28, units='inch')
    prob.set_val(Aircraft.HorizontalTail.WETTED_AREA, 0.352, units='m**2')

    prob.set_val(Aircraft.Wing.ROOT_CHORD, 20, units='inch')
    prob.set_val(Aircraft.Wing.SPAN, 4.667, units='ft')
    prob.set_val(Aircraft.Wing.WETTED_AREA, 0.85, units='m**2')

    prob.run_model()

    # -----------------------------
    # Output
    # -----------------------------
    print("\n===== DBF Mass Breakdown =====")
    print(f"Fuselage Mass: {prob.get_val(Aircraft.Fuselage.MASS)[0]:.4f} kg")
    print(f"Vertical Tail Mass: {prob.get_val(Aircraft.VerticalTail.MASS)[0]:.4f} kg")
    print(f"H-Tail Mass: {prob.get_val(Aircraft.HorizontalTail.MASS)[0]:.4f} kg")
    print(f"Wing Mass: {prob.get_val(Aircraft.Wing.MASS)[0]:.4f} kg")
    print(f"Battery Mass: {prob.get_val(Aircraft.Battery.MASS)[0]:.4f} kg")
    print(f"Motor Mass: {prob.get_val(Aircraft.Engine.Motor.MASS)[0]:.4f} kg")
    print(f"Total Mass: {prob.get_val('total_mass')[0]:.4f} kg")


if __name__ == '__main__':
    run_level3_dbf_example()
