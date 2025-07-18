import numpy as np
import os

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
    ribs = np.array([0]*14 + [1]*5 + [2])
    bulkhead_materials = np.where(ribs != 0, 'Ply', 'Balsa').tolist()
    rib_thicks = np.where(ribs == 2, 0.25, 0.125)

    fuselage = DBFFuselageMass()
    fuselage.options['bulkhead_materials'] = bulkhead_materials
    fuselage.options['bulkhead_thicknesses'] = (rib_thicks, 'inch')
    fuselage.options['num_spars'] = (0.5, 'unitless')
    fuselage.options['bulkhead_lightening_factor'] = (0.18, 'unitless')
    fuselage.options['sheeting_coverage'] = (1, 'unitless')
    fuselage.options['sheeting_density'] = (160, 'kg/m**3')
    fuselage.options['sheeting_lightening_factor'] = (0.3, 'unitless')
    fuselage.options['sheeting_thickness'] = (0.03125, 'inch')
    fuselage.options['glue_factor'] = (0.08, 'unitless')
    fuselage.options['stringer_density'] = (160, 'kg/m**3')
    fuselage.options['stringer_thickness'] = (0.375, 'inch')
    fuselage.options['floor_length'] = (2, 'ft')
    fuselage.options['floor_density'] = (340, 'kg/m**3')
    fuselage.options['floor_thickness'] = (0.125, 'inch')
    fuselage.options['skin_density'] = (20, 'g/m**2')
    fuselage.options['spar_density'] = (2, 'g/cm**3')
    fuselage.options['spar_outer_diameter'] = (1, 'inch')
    fuselage.options['spar_wall_thickness'] = (0.0625, 'inch')
    model.add_subsystem('fuselage', fuselage, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Configure vertical tail
    # -----------------------------
    vtail = DBFVerticalTailMass()
    rib_thicks_v = np.array([0.125]*5)
    rib_materials_v = ['Balsa'] * 4 + ['Ply'] * 1
    vtail.options['rib_materials'] = rib_materials_v
    vtail.options['rib_thicknesses'] = (rib_thicks_v, 'inch')
    vtail.options['airfoil_data_file'] = os.path.join(
        'aviary', 'examples', 'external_subsystems', 'dbf_based_mass', 'n0012-il.csv'
    )
    vtail.options['sheeting_coverage'] = (0.7, 'unitless')
    vtail.options['sheeting_density'] = (160, 'kg/m**3')
    vtail.options['sheeting_lightening_factor'] = (1.0, 'unitless')
    vtail.options['sheeting_thickness'] = (0.03125, 'inch')
    vtail.options['stringer_density'] = (160, 'kg/m**3')
    vtail.options['stringer_thickness'] = (0.375, 'inch')
    vtail.options['num_stringers'] = (2.5, 'unitless')
    vtail.options['glue_factor'] = (0.05, 'unitless')
    vtail.options['num_spars'] = (0, 'unitless')
    vtail.options['rib_lightening_factor'] = (2/3, 'unitless')
    vtail.options['skin_density'] = (20, 'g/m**2')
    vtail.options['spar_density'] = (0, 'g/cm**3')
    vtail.options['spar_outer_diameter'] = (0, 'inch')
    vtail.options['spar_wall_thickness'] = (0.0, 'inch')
    model.add_subsystem('vtail', vtail, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Configure horizontal tail
    # -----------------------------
    htail = DBFHorizontalTailMass()
    rib_thicks_h = np.array([0.125]*8)
    rib_materials_h = ['Balsa'] * 6 + ['Ply'] * 2
    htail.options['rib_materials'] = rib_materials_h
    htail.options['rib_thicknesses'] = (rib_thicks_h, 'inch')
    htail.options['airfoil_data_file'] = os.path.join(
        'aviary', 'examples', 'external_subsystems', 'dbf_based_mass', 'n0012-il.csv'
    )
    htail.options['sheeting_coverage'] = (0.7, 'unitless')
    htail.options['sheeting_density'] = (160, 'kg/m**3')
    htail.options['sheeting_lightening_factor'] = (1.0, 'unitless')
    htail.options['sheeting_thickness'] = (0.03125, 'inch')
    htail.options['stringer_density'] = (160, 'kg/m**3')
    htail.options['stringer_thickness'] = (0.375, 'inch')
    htail.options['num_stringers'] = (2.5, 'unitless')
    htail.options['glue_factor'] = (0.05, 'unitless')
    htail.options['num_spars'] = (0, 'unitless')
    htail.options['rib_lightening_factor'] = (2/3, 'unitless')
    htail.options['skin_density'] = (20, 'g/m**2')
    htail.options['spar_density'] = (0, 'g/cm**3')
    htail.options['spar_outer_diameter'] = (0, 'inch')
    htail.options['spar_wall_thickness'] = (0, 'inch')
    model.add_subsystem('htail', htail, promotes_inputs=['*'], promotes_outputs=['*'])

    # -----------------------------
    # Configure wing
    # -----------------------------
    wing = DBFWingMass()
    rib_thicks_w = np.array([0.125]*20)
    rib_materials_w = ['Balsa'] * 15 + ['Ply'] * 5
    wing.options['rib_materials'] = rib_materials_w
    wing.options['rib_thicknesses'] = (rib_thicks_w, 'inch')
    wing.options['airfoil_data_file'] = os.path.join(
        'aviary', 'examples', 'external_subsystems', 'dbf_based_mass', 'mh84-il.csv'
    )
    wing.options['sheeting_coverage'] = (0.4, 'unitless')
    wing.options['sheeting_density'] = (160, 'kg/m**3')
    wing.options['sheeting_lightening_factor'] = (1.0, 'unitless')
    wing.options['sheeting_thickness'] = (0.03125, 'inch')
    wing.options['stringer_density'] = (160, 'kg/m**3')
    wing.options['stringer_thickness'] = (0.375, 'inch')
    wing.options['num_stringers'] = (2.5, 'unitless')
    wing.options['glue_factor'] = (0.08, 'unitless')
    wing.options['num_spars'] = (1.1, 'unitless')
    wing.options['rib_lightening_factor'] = (2/3, 'unitless')
    wing.options['skin_density'] = (20, 'g/m**2')
    wing.options['spar_density'] = (2, 'g/cm**3')
    wing.options['spar_outer_diameter'] = (1, 'inch')
    wing.options['spar_wall_thickness'] = (0.0625, 'inch')
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
    print(f"Total Mass: {prob.get_val('structure_mass')[0]:.4f} kg")


if __name__ == '__main__':
    run_level3_dbf_example()
