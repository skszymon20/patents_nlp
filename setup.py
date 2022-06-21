from setuptools import setup, find_packages

setup(
    name='patents_nlp',
    version='0.0.1',
    description='Setting up a python package',
    author='Szymon Skwarek',
    author_email='skszymon20@gmail.com',
    packages=find_packages(include=['patents_nlp', 'patents_nlp.*']),
    install_requires=[
        'transformers==4.18.0',
        'pandas==1.4.2',
        'flake8==4.0.1',
        'mypy==0.950',
        'pycodestyle',
        'wandb'
    ],
    setup_requires=['pycodestyle'],
    tests_require=['pytest'],
)
# https://godatadriven.com/blog/a-practical-guide-to-using-setup-py/
