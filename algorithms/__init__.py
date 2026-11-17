import os
import sys

_alg_dir = os.path.dirname(os.path.abspath(__file__))
if _alg_dir not in sys.path:
    sys.path.insert(0, _alg_dir)
