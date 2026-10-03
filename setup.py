import re
from pathlib import Path

from setuptools import setup, find_packages

DESCRIPTION = "A trading framework for cryptocurrencies"
BASE_DIR = Path(__file__).resolve().parent

# The version lives only in algorithex/version.py. It is read as text because importing the package
# would import Algorithex and its dependencies during the build.
_version_match = re.search(
    r"^__version__\s*=\s*['\"]([^'\"]+)['\"]",
    (BASE_DIR / "algorithex" / "version.py").read_text(encoding="utf-8"),
    re.MULTILINE,
)
if not _version_match:
    raise RuntimeError("Could not find __version__ in algorithex/version.py")
VERSION = _version_match.group(1)

with open(BASE_DIR / "requirements.txt", "r", encoding="utf-8") as f:
    REQUIRED_PACKAGES = f.read().splitlines()

with open(BASE_DIR / "README.md", "r", encoding="utf-8") as f:
    LONG_DESCRIPTION = f.read()

# Algorithex is a self-contained distribution of this trading framework.
# The Python package path is `algorithex` (strategies import `from algorithex...`).
# Original MIT copyright (see LICENSE) is preserved there; this file carries product metadata only.
setup(
    name='algorithex',
    version=VERSION,
    author="Algorithex",
    author_email="support@algorithex.local",
    packages=find_packages(),
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="http://localhost:9000",
    project_urls={
        'Documentation': 'http://localhost:9000/docs',
        'Source': 'https://github.com/algorithex/algorithex',
        'Tracker': 'https://github.com/algorithex/algorithex/issues',
    },
    install_requires=REQUIRED_PACKAGES,
    entry_points='''
        [console_scripts]
        algorithex=algorithex.__init__:cli
    ''',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    include_package_data=True,
    package_data={
        '': ['*.dll', '*.dylib', '*.so', '*.json'],
    },
)
