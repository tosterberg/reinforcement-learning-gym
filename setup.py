from setuptools import setup, find_packages

setup(
    name="reinforce",
    version="0.0.1",
    author="Tyler Osterberg",
    author_email="tylertosterberg@gmail.com",
    description="Algorithms for Multi-Armed Bandits",
    url="https://github.com/tosterberg/reinforcement-learning-gym",
    packages=[package for package in find_packages()
              if package.startswith('reinforce')],
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'seaborn', 'pymc3'
    ]
)