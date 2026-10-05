import numpy as np
import datetime



def convert_to_string(input):
    input = str(input)
    return input


def standardize_domain(string: str):
    """
    Performs standardizations of strings to a common
    naming convention.

    Args:
        string (str):

    Returns:
        str
    """
    string = string.upper()
    if string == 'SURFACE':
        string = 'SEA_SURFACE'
    if string == 'SUBSURFACE':
        string = 'SEA_SUBSURFACE'
    if not (string in ['AIR', 'SPACE', 'GROUND', 'SEA_SURFACE', 'SEA_SUBSURFACE']):
        return_string = 'UNKNOWN'
    else:
        return_string = string
    return return_string

def alt_ge_0(alt: float):
    """
    Ensures altitude is greater than or equal to 0.

    Args:
        alt (float):

    Returns:
        alt (float)
    """
    if alt < 0:
        return 0
    else:
        return alt
    
def ft_to_m(alt: float):
    """
    Converts feet to meters for altitude.

    Args:
        alt (int):

    Returns:

    """
    return float(alt) * 0.3048

def knots_to_mps(speed: float):
    """Converts knots to meters per second."""
    return float(speed) * 0.514444

def convert_to_float(value):
    """
    Converts a value to a float.

    Args:
        value:

    Returns:
        float
    """
    try:
        return float(value)
    except:
        return np.nan

def degrees_to_radians(coord):
    """
    Converts degrees to radians.

    Args:
        coord:

    Returns:
        float
    """
    coord = np.radians(float(coord))
    return coord


def ms_to_ISO8601(timestamp):
    """
    Converts a timestamp from datetime to ISO8601 format.

    Args:
        timestamp:

    Returns:
        str
    """
    timestamp = float(timestamp) / 1000.
    timestamp = datetime.datetime.utcfromtimestamp(timestamp).isoformat(timespec='milliseconds') + 'Z'
    return timestamp


def sec_to_ISO8601(timestamp):
    """
    Converts a timestamp of seconds to ISO8601 format.
    Args:
        timestamp:

    Returns:
        str
    """
    return ms_to_ISO8601(float(timestamp) * 1000)