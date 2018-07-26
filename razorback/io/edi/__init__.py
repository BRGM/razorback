""" read/write MT proc data from/to edi files

See the edi.core module for tools working on the generic edi file format.
"""

from .mt_edi import load_edi, write_edi


__all__ = ['load_edi', 'write_edi']
