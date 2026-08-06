from enum import Enum, IntEnum, auto, unique

class WingType(Enum):
    '''
    Specifies the type of wing used in the UAV mass wing model computation.
    
    SIMPLE:
        Wing is made of solid foam and two hollow spars
    MEDIUM: 
        Wing design includes spars, sheeting, stringers, ribs, and is hollow
    '''

    SIMPLE = 'simple'
    MEDIUM = 'medium'
