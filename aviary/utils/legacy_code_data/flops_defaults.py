from aviary.utils.named_values import NamedValues

## Defaults ##
flops_default_values = NamedValues(
    {
        'WTIN.EEXP': (1.15, 'unitless'),
        'WTIN.IALTWT': (False, 'unitless'),
        'WTIN.CARGF': (False, 'unitless'),
        'WTIN.IFUFU': (False, 'unitless'),
        'WTIN.HYDPR': (3000.0, 'psi'),
        'WTIN.ULF': (3.75, 'unitless'),
        'WTIN.WPPASS': (165.0, 'lbm'),
        'ENGDIN.IDLE': (False, 'unitless'),
        'ENGDIN.IGEO': (False, 'unitless'),
        'ENGDIN.NONEG': (False, 'unitless'),
        'AERIN.MIKE': (False, 'unitless'),
        'AERIN.SWETF': (1, 'unitless'),
        'AERIN.SWETV': (1, 'unitless'),
        'FUSEIN.SWPLE': (45.0, 'deg'),
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
