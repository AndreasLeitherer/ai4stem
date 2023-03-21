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
    package_data={'ai4stem': ['data/experimental_images/*.npy',
                              'data/nn_predictions/*.npy',
                              'data/pretrained_models/*.h5,
                              'data/reference_images/*.npy']}
)
