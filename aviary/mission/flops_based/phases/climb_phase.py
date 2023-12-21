from aviary.mission.flops_based.phases.energy_phase import EnergyPhase, register


Climb = None  # forward declaration for type hints


@register
class Climb(EnergyPhase):
    '''
    Define a phase builder class for a typical FLOPS based climb phase.

    Attributes
    ----------
    name : str ('climb')
        object label

    aero_builder (None)
        utility for building and connecting a dynamic aerodynamics analysis component

    user_options : AviaryValues (<empty>)
        state/path constraint values and flags

        supported options:
            - num_segments : int (5)
                transcription: number of segments
            - order : int (3)
                transcription: order of the state transcription; the order of the control
                transcription is `order - 1`
            - fix_initial : bool (True)
            - fix_initial_time : bool (None)
            - initial_ref : float (1.0, 's')
                note: applied if, and only if, "fix_initial_time" is unspecified
            - initial_bounds : pair ((0.0, 100.0) 's')
                note: applied if, and only if, "fix_initial_time" is unspecified
            - duration_ref : float (1.0, 's')
            - duration_bounds : pair ((0.0, 100.0) 's')
            - initial_altitude : float (0.0, 'ft)
                starting true altitude from mean sea level
            - final_altitude : float
                ending true altitude from mean sea level
            - initial_mach : float (0.0, 'ft)
                starting Mach number
            - final_mach : float
                ending Mach number
            - required_altitude_rate : float (None)
                minimum avaliable climb rate
            - no_climb : bool (False)
                aircraft is not allowed to climb during phase
            - no_descent : bool (False)
                aircraft is not allowed to descend during phase
            - fix_range : bool (False)
            - input_initial : bool (False)

    initial_guesses : AviaryValues (<empty>)
        state/path beginning values to be set on the problem

        supported options:
            - times
            - range
            - altitude
            - velocity
            - velocity_rate
            - mass
            - throttle

    ode_class : type (None)
        advanced: the type of system defining the ODE

    transcription : "Dymos transcription object" (None)
        advanced: an object providing the transcription technique of the
        optimal control problem

    external_subsystems : Sequence["subsystem builder"] (<empty>)
        advanced

    meta_data : dict (<"builtin" meta data>)
        advanced: meta data associated with variables in the Aviary data hierarchy

    default_name : str
        class attribute: derived type customization point; the default value
        for name

    default_ode_class : type
        class attribute: derived type customization point; the default value
        for ode_class used by build_phase

    default_meta_data : dict
        class attribute: derived type customization point; the default value for
        meta_data

    Methods
    -------
    build_phase
    make_default_transcription
    validate_options
    assign_default_options
    '''
    # region : derived type customization points
    __slots__ = ()

    default_name = 'climb'
    # endregion : derived type customization points
