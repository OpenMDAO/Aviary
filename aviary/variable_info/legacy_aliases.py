"""Compatibility aliases for renamed Aviary variable keys.

New code and generated data should always use canonical names. This map is
only for reading or accessing values that were written with an older Aviary
variable name.
"""

LEGACY_VARIABLE_NAME_MAP = {
    'mission:constraints:max_mach': 'aircraft:design:max_mach',
}


def resolve_legacy_variable_name(name):
    """Return the canonical name for a supported legacy Aviary variable."""
    return LEGACY_VARIABLE_NAME_MAP.get(name, name)
