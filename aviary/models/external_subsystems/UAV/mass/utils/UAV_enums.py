from enum import Enum


class UAVWingType(Enum):
    """
    Specifies the type of wing used in the UAV mass wing model computation.

    SOLID:
        Wing is made of solid foam and two hollow spars
    HOLLOW:
        Wing design includes spars, sheeting, stringers, ribs, and is hollow
    """

    SOLID = 'solid'
    HOLLOW = 'hollow'
