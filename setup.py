from setuptools import setup, find_packages

setup(
    name='QM',
    version='0.1.0',
    description='Quantum',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'xgboost==2.1.0',
        'numpy==1.26.4',
        'scikit-learn==1.5.0',
        'scipy==1.14.0',
        'networkx==3.3',
        'torch==2.3.1',
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
