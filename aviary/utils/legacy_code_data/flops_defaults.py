from aviary.utils.named_values import NamedValues

## Defaults ##
flops_default_values = NamedValues(
    {
        'WTIN.EEXP': (1.15, 'unitless'),
        'WTIN.IALTWT': (False, 'unitless'),
        'WTIN.CARGF': (False, 'unitless'),
        'WTIN.IFUFU': (False, 'unitless'),
        'ENGDIN.IDLE': (False, 'unitless'),
        'ENGDIN.IGEO': (False, 'unitless'),
        'ENGDIN.NONEG': (False, 'unitless'),
        'AERIN.MIKE': (False, 'unitless'),
        'AERIN.SWETF': (1, 'unitless'),
        'AERIN.SWETV': (1, 'unitless'),
    }
)

## Depreciated Variables ##
flops_deprecated_vars = [
    'AERIN.MODARO',
    'ENGDIN.IGENEN',
    'ENGDIN.BOOST',
    'ENGDIN.EXTFAC',
    'ENGDIN.IXTRAP',
    'ENGDIN.NPCODE',
    'ENGDIN.PCODE',
    'ENGDIN.NOX',
    'OPTION.IOPT',
    'OPTION.IPOLP',
    'OPTION.NOISE',
    'WTIN.CARBAS',
]
