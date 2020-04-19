import os
import sys
from pathlib import Path

MITSUBA_BASE = Path('/home/christian/github/mitsuba')

for path in (MITSUBA_BASE / 'dist/python').iterdir():
    if str(path) not in sys.path:
        sys.path.append(str(path.absolute()))

if str(MITSUBA_BASE / 'dist') not in os.environ['PATH']:
    os.environ['PATH'] = str(MITSUBA_BASE / 'dist') + os.pathsep + os.environ['PATH']
