#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
sRGB Colourspace
================

Defines the *sRGB* colourspace:

-   :attr:`sRGB_COLOURSPACE`.

See Also
--------
`RGB Colourspaces Jupyter Notebook
<http://nbviewer.jupyter.org/github/colour-science/colour-notebooks/\
blob/master/notebooks/models/rgb.ipynb>`_

References
----------
.. [1]  International Electrotechnical Commission. (1999). IEC 61966-2-1:1999 -
        Multimedia systems and equipment - Colour measurement and management -
        Part 2-1: Colour management - Default RGB colour space - sRGB, 51.
        Retrieved from https://webstore.iec.ch/publication/6169
.. [2]  International Telecommunication Union. (2015). Recommendation
        ITU-R BT.709-6 - Parameter values for the HDTV standards for production
        and international programme exchange BT Series Broadcasting service.
        Retrieved from https://www.itu.int/dms_pubrec/itu-r/rec/bt/\
R-REC-BT.709-6-201506-I!!PDF-E.pdf
"""

from __future__ import division, unicode_literals

import numpy as np

from colour.colorimetry import ILLUMINANTS
from colour.models.rgb import (RGB_Colourspace, oetf_sRGB, eotf_sRGB)

__author__ = 'Colour Developers'
__copyright__ = 'Copyright (C) 2013-2017 - Colour Developers'
__license__ = 'New BSD License - http://opensource.org/licenses/BSD-3-Clause'
__maintainer__ = 'Colour Developers'
__email__ = 'colour-science@googlegroups.com'
__status__ = 'Production'

__all__ = [
    'sRGB_PRIMARIES', 'sRGB_ILLUMINANT', 'sRGB_WHITEPOINT',
    'sRGB_TO_XYZ_MATRIX', 'XYZ_TO_sRGB_MATRIX', 'sRGB_COLOURSPACE'
]

sRGB_PRIMARIES = np.array(
    [[0.6400, 0.3300],
     [0.3000, 0.6000],
     [0.1500, 0.0600]])  # yapf: disable
"""
*sRGB* colourspace primaries.

sRGB_PRIMARIES : ndarray, (3, 2)
"""

sRGB_ILLUMINANT = 'D65'
"""
*sRGB* colourspace whitepoint name as illuminant.

sRGB_WHITEPOINT : unicode
"""

sRGB_WHITEPOINT = (
    ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][sRGB_ILLUMINANT])
"""
*sRGB* colourspace whitepoint.

sRGB_WHITEPOINT : ndarray
"""

sRGB_TO_XYZ_MATRIX = np.array(
    [[0.4124, 0.3576, 0.1805],
     [0.2126, 0.7152, 0.0722],
     [0.0193, 0.1192, 0.9505]])  # yapf: disable
"""
*sRGB* colourspace to *CIE XYZ* tristimulus values matrix.

sRGB_TO_XYZ_MATRIX : array_like, (3, 3)
"""

XYZ_TO_sRGB_MATRIX = np.array(
    [[3.2406, -1.5372, -0.4986],
     [-0.9689, 1.8758, 0.0415],
     [0.0557, -0.2040, 1.0570]])  # yapf: disable
"""
*CIE XYZ* tristimulus values to *sRGB* colourspace matrix.

XYZ_TO_sRGB_MATRIX : array_like, (3, 3)
"""

sRGB_COLOURSPACE = RGB_Colourspace(
    'sRGB',
    sRGB_PRIMARIES,
    sRGB_WHITEPOINT,
    sRGB_ILLUMINANT,
    sRGB_TO_XYZ_MATRIX,
    XYZ_TO_sRGB_MATRIX,
    oetf_sRGB,
    eotf_sRGB)  # yapf: disable
"""
*sRGB* colourspace.

sRGB_COLOURSPACE : RGB_Colourspace
"""
