""" packaging information for ipyml
"""

import re
import sys
from pathlib import Path

import setuptools

if __name__ == "__main__":
    setuptools.setup(
        version=re.findall(
            r"""__version__ = "([^"]+)"$""",
            (Path(__file__).parent / "src" / "ipyml" / "_version.py").read_text(),
        )[0],
    ) 