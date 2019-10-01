import sys
from setuptools import setup

install_requires = [
    'Click',
    'enum34',
    'pyDOE',
    'hyperopt',
    'sklearn'
]
if sys.version_info > (3, 0):
    install_requires.append('numpy')
    install_requires.append('scipy')
    install_requires.append('pytest == 4.6.5')
    install_requires.append('scikit-learn')
else:
    install_requires.append('numpy == 1.15')
    install_requires.append('scipy == 0.16')
    install_requires.append('pytest == 4.6.5')
    install_requires.append('scikit-learn == 0.20.4')

setup(
    name="uchit",
    version='0.1',
    py_modules=['uchit'],
    install_requires=install_requires,
    entry_points='''
        [console_scripts]
        uchit=uchit:start
    ''',
)
