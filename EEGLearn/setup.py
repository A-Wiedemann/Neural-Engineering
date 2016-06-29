from distutils.core import setup

setup(
    name='EEGLearn',
    version='1',
    packages=['EEGLearn'],
    install_requires=['numpy', 'scipy', 'scikit-learn', 'theano', 'lasagne'],
    url='https://github.com/pbashivan/EEGLearn',
    license='GNU GENERAL PUBLIC LICENSE',
    author='Pouya Bashivan',
    author_email='poya.bashivan@gmail.com',
    description='Representation learning from EEG'
)
