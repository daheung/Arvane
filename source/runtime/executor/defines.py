from enum import Enum, IntFlag

class EProcessStatus(IntFlag):
    IDLE    = 0x0000
    RUNNING = 0x0001
    WAIT    = 0x0002

    PENDING_KILL = 0x000F
    
    ABORTED = 0xF000

class EProcessType(Enum):
    DEPTH   = 0x0001
    RECON   = 0x0002
    EXTRACT = 0x0004

    NOT_SET = 0x00FF