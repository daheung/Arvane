from enum import Enum, IntEnum

class TaskStatus(IntEnum):
    # Define task status Success codes
    SUCCESS = 0

    # Define task status Error codes
    ABORTED = -1
    DONE_WITH_ERROR = -2

    # Define task status Warning codes
    WARNING = 0xA001

    # Define task status Info codes
    NOT_MODIFIED = 0xB001