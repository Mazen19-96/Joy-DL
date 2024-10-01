from setuptools import setup

setup (
    name='Joy',
    version='1.0',
    description='Joy is a simple deeplearning library written in python to creat and training Neural Networks ',
    author='Mazen Saleh',
    author_email='Rmen1996@gmail.com',
    packages=['Joy','example'],
    install_requires=['numpy>=1.16.2'],
    keywords=['Automatic differentiation', 'backpropagation', 'deep learning'
             'machine learning', 'optimization', 'neural networks',
              'Python', 'Numpy'],
    url='https://github.com/Mazen19-96/Joy',
    classifiers=['Programming Language :: Python :: 3.7']
    )  