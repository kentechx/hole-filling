from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

# https://stackoverflow.com/a/7071358/148668
import re

VERSIONFILE = "hole_filling/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name='hole-filling',
    version=verstr,
    author='Kaidi Shen',
    description='A simple Python package for hole filling in triangle meshes.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/kentechx/hole-filling',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
    install_requires=['numpy<2.0.0', 'libigl==2.4.1', 'scipy'],
)
