# -*- coding: utf-8 -*-

"""Top-level package for lidar (trimmed vendor copy - see lidar_pkg note below).

Only ExtractSinks (sink/depression extraction) and MedianFilter (DEM
smoothing) are used by this pipeline; the rest of the upstream `lidar`
package (GUI, mounts/peak delineation, ArcGIS toolbox, docs, examples)
was removed to cut vendored bloat. Original project:
https://github.com/opengeos/lidar
"""

__author__ = """Qiusheng Wu"""
__email__ = "giswqs@gmail.com"
__version__ = "0.8.4"

from .filling import ExtractSinks
from .filtering import MedianFilter
