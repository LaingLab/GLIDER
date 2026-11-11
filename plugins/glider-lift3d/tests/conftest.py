import sys
from pathlib import Path

# The arch tests reuse test_lift3d's synthetic mouse and stand-in models.
sys.path.insert(0, str(Path(__file__).parent))
