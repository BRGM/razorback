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


from .signalset import *
