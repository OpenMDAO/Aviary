thickness_to_chord_scaler = 0.387
_fuselage_adjustment_scaler = 0.6730


def calc_lifting_surface_scaler(thickness_to_chord):
    """Calculate lifting surface scaler."""
    scaler = thickness_to_chord_scaler * thickness_to_chord + 2.0

    return scaler


def calc_fuselage_adjustment(chord, thickness_to_chord):
    """Calculate fuselage_adjustment."""
    adjustment = \
        _fuselage_adjustment_scaler * chord * (thickness_to_chord * chord)

    return adjustment


def d_calc_fuselage_adjustment(chord, thickness_to_chord):
    """
    Calculate partial derivatives of fuselage_adjustment with respect to chord and thickness_to_chord.
    """
    d1 = 2.0 * _fuselage_adjustment_scaler * thickness_to_chord * chord
    d2 = _fuselage_adjustment_scaler * chord ** 2

    return d1, d2


class Names:
    '''
    Define component I/O variable names that should not exported.
    '''

    CROOT = 'prep_geom:_Names:CROOT'
    CROOTB = 'prep_geom:_Names:CROOTB'
    CROTM = 'prep_geom:_Names:CROTM'
    CROTVT = 'prep_geom:_Names:CROTVT'
    CRTHTB = 'prep_geom:_Names:CRTHTB'
    # equivalent(?)
    SPANHT = 'prep_geom:_Names:SPANHT'
    SPANVT = 'prep_geom:_Names:SPANVT'
    XDX = 'prep_geom:_Names:XDX'
    XMULT = 'prep_geom:_Names:XMULT'
    XMULTH = 'prep_geom:_Names:XMULTH'
    XMULTV = 'prep_geom:_Names:XMULTV'

    __slots__ = ()
