import numpy as np
import openmdao.api as om

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.variables import Aircraft, Dynamic, Mission


def generate_drag_report(prob):
    """
    Reproduces the FLOPS drag report for the given Aviary model.

    This is a legacy report that can be useful when comparing with the FLOPS output file.

    Parameters
    ----------
    prob : AivaryProblem
        The problem to report.

    Returns
    -------
    str
        Report content.
    """
    # Pull builder from the problem.
    aero_builder = prob.model.subsystem_builders[4]

    aero_mission = aero_builder.build_mission(
        num_nodes=1,
        aviary_inputs=prob.aviary_inputs,
        user_options={},
        subsystem_options={'method': 'computed'},
    )
    params = aero_builder.get_parameters(
        aviary_inputs=prob.aviary_inputs,
        user_options={},
        subsystem_options={'method': 'computed'},
    )

    aero_problem = om.Problem(reports=False)
    model = aero_problem.model
    model.add_subsystem(
        'atmosphere',
        Atmosphere(num_nodes=1),
        promotes_inputs=[Dynamic.Mission.ALTITUDE],
        promotes_outputs=[
            Dynamic.Atmosphere.DENSITY,
            Dynamic.Atmosphere.SPEED_OF_SOUND,
            Dynamic.Atmosphere.TEMPERATURE,
            Dynamic.Atmosphere.STATIC_PRESSURE,
            Dynamic.Atmosphere.DYNAMIC_VISCOSITY,
        ],
    )
    model.add_subsystem('aero', aero_mission, promotes=['*'])

    setup_model_options(aero_problem, prob.aviary_inputs, prob.meta_data)

    aero_problem.setup()
    aero_problem.set_solver_print(0)

    # Set hieararchy variables
    for var, meta in params.items():
        val = prob.get_val(var, units=meta['units'])
        aero_mission.set_val(var, val, units=meta['units'])

    # Mimic the EDET report

    lines = '## AERODYNAMIC DATA\n'
    lines += '```\n'

    val = prob.get_val(Aircraft.Wing.SPAN_EFFICIENCY_FACTOR, units='unitless')[0]
    lines += f' SPAN EFFICIENCY FACTOR =   {val:.5f}\n\n'

    lines += ' MACH NUMBER       ALTITUDE     REFERENCE AREA     TECHNOLOGY LEVEL\n'
    mach = prob.get_val(Aircraft.Design.CRUISE_MACH, units='unitless')[0]
    cruise_alt = prob.aviary_inputs.get_val(Aircraft.Design.CRUISE_ALTITUDE, units='ft')
    area = prob.get_val(Aircraft.Wing.AREA, units='ft**2')[0]
    tec = prob.aviary_inputs.get_val(Aircraft.Wing.AIRFOIL_TECHNOLOGY, units='unitless')
    lines += f'     {mach:.2f}     {cruise_alt:>9.0f}. FT'
    lines += f'    {area:>9.2f} SQ FT           {tec:.2f}\n\n'

    # Run the design mission point
    aero_problem.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
    aero_problem.set_val(Dynamic.Mission.ALTITUDE, cruise_alt, units='ft')
    aero_problem.run_model()

    lines += '   COMPONENT              SWET      LENGTH    FINENESS     FORM       RN'
    lines += '       CF         CDF\n'
    lines += 26 * ' ' + 'SQ FT       FT        RATIO     FACTOR   MILLIONS\n'

    nvtail = prob.aviary_inputs.get_val(Aircraft.VerticalTail.NUM_TAILS)
    try:
        nhtail = prob.aviary_inputs.get_val(Aircraft.HorizontalTail.NUM_TAILS)
    except:
        # component-wise default.
        nhtail = 1

    nfuse = prob.aviary_inputs.get_val(Aircraft.Fuselage.NUM_FUSELAGES)
    num_engines = prob.aviary_inputs.get_val(Aircraft.Engine.NUM_ENGINES)

    form = aero_problem.model.aero.SkinFrictionDrag.form_factor
    rn = aero_problem.get_val('Re', units='unitless') / 1.0e6
    cf = aero_problem.get_val('skin_friction_coeff', units='unitless')
    cdf = aero_problem.model.aero.SkinFrictionDrag.cdf_per_surface

    # Skin Friction: Wing
    j = 0
    swet = aero_problem.get_val(Aircraft.Wing.WETTED_AREA, units='ft**2')[0]
    length = aero_problem.get_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, units='ft')[0]
    fine = aero_problem.get_val(Aircraft.Wing.FINENESS, units='unitless')[0]
    lines += f' WING            {swet:>14.2f} {length:>10.2f} {fine:>10.4f} {form[j]:>10.3f} '
    lines += f'{rn[0, j]:>8.1f} {cf[0, j]:>10.5f} {cdf[0, j]:>10.5f}\n'

    # Skin Friction: Horizontal Tail
    for ii in range(nhtail):
        j += 1
        swet = aero_problem.get_val(Aircraft.HorizontalTail.WETTED_AREA, units='ft**2')[0]
        length = aero_problem.get_val(Aircraft.HorizontalTail.CHARACTERISTIC_LENGTH, units='ft')[0]
        fine = aero_problem.get_val(Aircraft.HorizontalTail.FINENESS, units='unitless')[0]
        lines += f' HORIZONTAL TAIL {swet:>14.2f} {length:>10.2f} {fine:>10.4f} {form[j]:>10.3f} '
        lines += f'{rn[0, j]:>8.1f} {cf[0, j]:>10.5f} {cdf[0, j]:>10.5f}\n'

    # Skin Friction: Vertical Tail
    for ii in range(nvtail):
        j += 1
        swet = aero_problem.get_val(Aircraft.VerticalTail.WETTED_AREA, units='ft**2')[0]
        length = aero_problem.get_val(Aircraft.VerticalTail.CHARACTERISTIC_LENGTH, units='ft')[0]
        fine = aero_problem.get_val(Aircraft.VerticalTail.FINENESS, units='unitless')[0]
        lines += f' VERTICAL TAIL   {swet:>14.2f} {length:>10.2f} {fine:>10.4f} {form[j]:>10.3f} '
        lines += f'{rn[0, j]:>8.1f} {cf[0, j]:>10.5f} {cdf[0, j]:>10.5f}\n'

    # Skin Friction: Fuselage
    for ii in range(nfuse):
        j += 1
        swet = aero_problem.get_val(Aircraft.Fuselage.WETTED_AREA, units='ft**2')[0]
        length = aero_problem.get_val(Aircraft.Fuselage.CHARACTERISTIC_LENGTH, units='ft')[0]
        fine = aero_problem.get_val(Aircraft.Fuselage.FINENESS, units='unitless')[0]
        lines += f' FUSELAGE        {swet:>14.2f} {length:>10.2f} {fine:>10.4f} {form[j]:>10.3f} '
        lines += f'{rn[0, j]:>8.1f} {cf[0, j]:>10.5f} {cdf[0, j]:>10.5f}\n'

    # Skin Friction: Nacelles
    for num_this_type in num_engines:
        for jj in range(num_this_type):
            j += 1
            swet = aero_problem.get_val(Aircraft.Nacelle.WETTED_AREA, units='ft**2')[ii]
            length = aero_problem.get_val(Aircraft.Nacelle.CHARACTERISTIC_LENGTH, units='ft')[ii]
            fine = aero_problem.get_val(Aircraft.Nacelle.FINENESS, units='unitless')[ii]
            lines += f' NACELLE         {swet:>14.2f} {length:>10.2f} {fine:>10.4f} '
            lines += f'{form[j]:>10.3f} {rn[0, j]:>8.1f} {cf[0, j]:>10.5f} {cdf[0, j]:>10.5f}\n'

    # Skin Friction Misc
    cdftot = aero_problem.get_val('SkinFrictionDrag.skin_friction_drag_coeff', units='unitless')[0]
    misc = cdftot - np.sum(cdf)
    lines += f' MISCELLANEOUS {misc:>80.5f}\n\n'

    # Total Skin Friction
    swtot = prob.get_val(Aircraft.Design.TOTAL_WETTED_AREA, units='ft**2')[0]
    lines += f'  TOTAL          {swtot:>14.2f} {cdftot:>63.5f}\n\n'

    # Compute Drag deltas over mach/alt sweep.
    alts = np.linspace(0, 60000, 13)
    machs = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.775, 0.8, 0.825, 0.850])

    lines += '       CHANGE IN DRAG COEFFICIENT FROM CRUISE ALTITUDE\n'
    lines += ' ALTITUDE                                 MACH NUMBER\n'

    line = '         '
    for mach in machs:
        line += f'{mach:>9.3f}'
    lines += line + '\n'

    base_CD = np.zeros(len(machs))
    qS = np.zeros(len(machs))
    for j, mach in enumerate(machs):
        aero_problem.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
        aero_problem.run_model()
        CD = aero_problem.get_val('SkinFrictionDrag.skin_friction_drag_coeff', units='unitless')[0]
        base_CD[j] = CD
        q = aero_problem.get_val('dynamic_pressure', units='lbf/ft**2')[0]
        qS[j] = q * area

    for alt in alts:
        aero_problem.set_val(Dynamic.Mission.ALTITUDE, alt, units='ft')
        line = f'{alt:>8.0f}. '

        for j, mach in enumerate(machs):
            aero_problem.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
            aero_problem.run_model()

            CD = aero_problem.get_val(
                'SkinFrictionDrag.skin_friction_drag_coeff',
                units='unitless',
            )[0]
            delta = CD - base_CD[j]
            line += f'{delta:>9.5f}'

        lines += line + '\n'

    # Drag Polar
    cls = np.linspace(0.2, 0.9, 15)
    aero_problem.set_val(Dynamic.Mission.ALTITUDE, cruise_alt, units='ft')

    # FLOPS computes the polar before scaling any drag components.
    aero_problem.set_val(Aircraft.Design.ZERO_LIFT_DRAG_COEFF_FACTOR, 1.0)
    aero_problem.set_val(Aircraft.Design.LIFT_DEPENDENT_DRAG_COEFF_FACTOR, 1.0)
    aero_problem.set_val(Aircraft.Design.SUBSONIC_DRAG_COEFF_FACTOR, 1.0)

    lines += f'\n           DRAG POLARS     ALTITUDE ={cruise_alt:>6.0f}. FT\n'
    lines += ' MACH                                                  CL\n'

    line = '      '
    for cl in cls:
        line += f'{cl:>8.3f}'
    lines += line + '\n'

    for j, mach in enumerate(machs):
        aero_problem.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
        line = f'{mach:>6.3f} '

        for cl in cls:
            mass = cl * qS[j]
            aero_problem.set_val(Dynamic.Vehicle.MASS, mass, units='lbm')
            aero_problem.run_model()

            CD = aero_problem.get_val('drag_coefficient', units='unitless')[0]
            line += f'{CD:>8.5f}'

        lines += line + '\n'

    # Drag Components
    des_mach = prob.get_val(Aircraft.Design.MACH, units='unitless')[0]
    des_cl = prob.get_val(Aircraft.Design.LIFT_COEFFICIENT, units='unitless')[0]
    lines += f'\n  DESIGN MACH NUMBER = {des_mach:>5.3f}, DESIGN CL = {des_cl:>6.3f}\n\n'

    lines += '       MACH               CDF                 CDC               BUFFET CL\n\n'
    aero_problem.set_val(Dynamic.Atmosphere.MACH, des_mach, units='unitless')
    for mach in machs:
        line = f'{mach:>11.3f} '

        aero_problem.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
        aero_problem.run_model()

        cdf = aero_problem.get_val(
            'SkinFrictionDrag.skin_friction_drag_coeff',
            units='unitless',
        )[0]
        line += f'{cdf:>19.5f} '

        cdc = aero_problem.get_val('CompressibilityDrag.compress_drag_coeff', units='unitless')[0]
        line += f'{cdc:>19.5f} '

        delbuf = aero_problem.get_val('Buffet.DELCLB', units='unitless')[0]
        buf = des_cl + delbuf
        line += f'{buf:>19.3f}'

        lines += line + '\n'

    lines += '\n        CL                CDI\n\n'
    aero_problem.set_val(Dynamic.Atmosphere.MACH, des_mach, units='unitless')
    aero_problem.run_model()
    q = aero_problem.get_val('dynamic_pressure', units='lbf/ft**2')[0]
    qS_des = q * area

    for j, cl in enumerate(cls):
        line = f'{cl:>11.3f} '

        mass = cl * qS_des
        aero_problem.set_val(Dynamic.Vehicle.MASS, mass, units='lbm')
        aero_problem.run_model()

        cdi = aero_problem.get_val('InducedDrag.induced_drag_coeff', units='unitless')[0]
        line += f'{cdi:>19.5f}'

        lines += line + '\n'

    # Pressure Drag
    lines += '\n           PRESSURE DRAG COEFFICIENTS...CDP\n'
    lines += ' MACH                                                  CL\n'

    line = '      '
    for cl in cls:
        line += f'{cl:>8.3f}'
    lines += line + '\n'

    for j, mach in enumerate(machs):
        aero_problem.set_val(Dynamic.Atmosphere.MACH, mach, units='unitless')
        line = f'{mach:>6.3f} '

        for cl in cls:
            mass = cl * qS[j]
            aero_problem.set_val(Dynamic.Vehicle.MASS, mass, units='lbm')
            aero_problem.run_model()

            CD = aero_problem.get_val('PressureDrag.pressure_drag_coeff', units='unitless')[0]
            line += f'{CD:>8.5f}'

        lines += line + '\n'

    lines += '```\n'
    return lines
