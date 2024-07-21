from setuptools import setup, find_packages

setup(
    name='QM',
    version='0.1.0',
    description='Quantum',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'xgboost==1.0.2',
        'numpy==1.16.4',
        'scikit-learn==0.20.3',
        'matplotlib==3.0.3',
        'seaborn==0.9.0',
        'scipy==1.2.1',
        'networkx==2.3',
        'torch==1.1.0',
    ],
    test_suite='tests',
    tests_require=[
        'unittest2',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
