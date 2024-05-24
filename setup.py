from setuptools import setup, find_namespace_packages
from json import load

install_requires = open('requirements.txt').read().split('\n')
with open('ocrd-tool.json', 'r', encoding='utf-8') as f:
    version = load(f)['version']

setup(
    name='eynollah',
    version=version,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vahid Rezanezhad',
    url='https://github.com/qurator-spk/eynollah',
    license='Apache License 2.0',
    packages=find_namespace_packages(include=['qurator']),
    install_requires=install_requires,
    package_data={
        '': ['*.json']
    },
    entry_points={
        'console_scripts': [
            'eynollah=qurator.eynollah.cli:main',
            'ocrd-eynollah-segment=qurator.eynollah.ocrd_cli:main',
        ]
    },
)
