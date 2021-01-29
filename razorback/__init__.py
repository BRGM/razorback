""" razorback: tools for robust estimations of transfer functions.
"""

__version__ = "0.3.0.4"

try:
    import razorback_plus as plus
except ModuleNotFoundError:
    plus = None


from .errors import *

from . import fourier_transform
from . import mestimator

from . import weights
from . import prefilters

from . import data

from . import calibrations
from . import io

from . import utils

from .signalset import *
