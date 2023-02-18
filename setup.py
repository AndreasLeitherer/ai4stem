from setuptools import setup, find_packages

setup(
    name='ai4stem',
    version='0.1',
    author='A. Leitherer, C. Liebscher, L.M. Ghiringhelli',
    author_email='leitherer@fhi-berlin.mpg.de',
    url='https://github.com/AndreasLeitherer/ai4stem/',
    description='AI analytics software for STEM',
    long_description='To be completed.',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib',
                      'seaborn', 'pandas', 'opencv-python',
                      'scikit-image'],
    package_data={'ai4stem': ['data/haadf_experimental_lattice_parameters.npy',
                              'data/haadf_experimental_lattice_parameters_labels.npy']}
)
