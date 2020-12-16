#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from setuptools import setup

install_requires = open('requirements.txt').read().split('\n')

setup(
    name='eynollah',
    version='0.0.1',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vahid Rezanezhad',
    url='https://github.com/qurator-spk/eynollah',
    license='Apache License 2.0',
    packages=['sbb_newspapers_org_image'],
    install_requires=install_requires,
    entry_points={
        'console_scripts': [
            'eynollah=sbb_newspapers_org_image.cli:main',
            # 'ocrd-eynollah=eynollah.ocrd_cli:cli',
        ]
    },
)
