# -*- coding: utf-8 -*-
from build import *
from setuptools import setup

packages = \
    ['fastms']

package_data = \
    {'': []}

install_requires = \
    ['numpy']

setup_kwargs = {
    'name': 'fastms',
    'version': '0.1',
    'description': 'fastms Python bindings',
    'long_description': 'fastms Python bindings',
    'author': 'Max Edell',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
}

build(setup_kwargs)

setup(**setup_kwargs)
