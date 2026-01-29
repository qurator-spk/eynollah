"""
Load libraries with possible race conditions once. This must be imported as the first module of eynollah.
"""
from ocrd_utils import tf_disable_interactive_logs
from torch import *
tf_disable_interactive_logs()
import tensorflow.keras
from shapely import *
imported_libs = True
__all__ = ['imported_libs']
