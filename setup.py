from setuptools import setup

setup(
    name="uchit",
    version='0.1',
    py_modules=['uchit'],
    install_requires=[
        'Click',
        'enum34',
        'pyDOE',
        'numpy'
    ],
    entry_points='''
        [console_scripts]
        uchit=uchit:start
    ''',
)