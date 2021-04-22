from setuptools import setup, find_packages

install_requires = open('requirements.txt').read().split('\n')

setup(
    name='eynollah',
    version='0.0.1',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Vahid Rezanezhad',
    url='https://github.com/qurator-spk/eynollah',
    license='Apache License 2.0',
    namespace_packages=['qurator'],
    packages=find_packages(exclude=['tests']),
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
