"""
Load libraries with possible race conditions once. This must be imported as the first module of eynollah.
"""
from torch import *
import tensorflow.keras
from shapely import *
imported_libs = True
__all__ = ['imported_libs']
