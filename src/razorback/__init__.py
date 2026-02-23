""" razorback: tools for robust estimations of transfer functions.
"""

try:
    import razorback_plus as plus
except ModuleNotFoundError:
    plus = None


try:
    __version__ = __import__('importlib.metadata').metadata.version(__name__)
except:
    __version__ = None


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
