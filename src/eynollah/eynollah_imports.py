"""
Load libraries with possible race conditions once. This must be imported as the first module of eynollah.
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1' # avoid Keras 3 after TF 2.15

from ocrd_utils import tf_disable_interactive_logs
from torch import *
tf_disable_interactive_logs()
import tensorflow.keras
from shapely import *
imported_libs = True
__all__ = ['imported_libs']
