import os

from setuptools import (
    setup,
    find_packages
    )

_CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

try:
    README = open(os.path.join(_CURRENT_DIR, "README.md"),
                  encoding="utf-8").read()
except IOError:
    README = ""

setup(
    name="emloss",
    version="0.0.1",
    description='PMSM core loss in JAX',
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/afbeltranp/emloss-paper",
    author="Andrés Beltrán-Pulido",
    author_email='beltranp@purdue.edu',
    license='Apache-2.0',
    packages=find_packages(),
    install_requires=[      
        'jax',
        'jaxlib',
        'jaxtyping' 
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache-2.0 license',  
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Physics',
    ],
)
