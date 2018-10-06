#!/usr/bin/env python

import io
import os
import sys
from shutil import rmtree

import setuptools as sutools

VERSION = '0.0.1'

# Package meta-data.
NAME = 'dsynth'
DESCRIPTION = 'dsynth generates synthetic scenes using multiview images of individual objects and physics simulation.'
URL = 'https://github.com/dnnspark/dsynth'
EMAIL = 'dnnsipark@gmail.com'
AUTHOR = 'Dennis Park'
REQUIRES_PYTHON = ">=2.7, !=3.0.*, !=3.1.*, !=3.2.*, !=3.3.*"  # 2.7 or >=3.4

# What packages are required for this module to be executed?
REQUIRED = [
    'numpy',
    'matplotlib',
    'imageio',
    'pyyaml',
]

# What packages are optional?
EXTRAS = {}

# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = '\n' + f.read()

# # Load the package's __version__.py module as a dictionary.

# Where the magic happens:
sutools.setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    # packages=sutools.find_packages(exclude=('tests',)),
    packages=[
        NAME,  # main src dir has same name as package
        'tests'
    ],
    # If your package is a single module, use this instead of 'packages':
    # py_modules=['mypackage'],

    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        # 'Programming Language :: Python',
        # 'Programming Language :: Python :: 3',
        # 'Programming Language :: Python :: 3.6',
        # 'Programming Language :: Python :: Implementation :: CPython',
        # 'Programming Language :: Python :: Implementation :: PyPy'
        'Development Status :: 2 - Pre-Alpha',
        # 'Intended Audience :: Developers',
        'Natural Language :: English',
        # 'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
    # $ setup.py publish support.
    cmdclass={
        # 'upload': UploadCommand,
    },

    zip_safe=False,
)
