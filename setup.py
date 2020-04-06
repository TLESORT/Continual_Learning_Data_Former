from setuptools import setup

with open("Readme.md", "r") as fh:

    long_description = fh.read()

setup(
    name='continuum',  
    version='0.1',
    author="Timothee LESORT",
    author_email="t.lesort@gmail.com",
    description="A data loader for continual learning experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="#",
    packages=['builders', 'builders.datasets'],
    zip_safe=False

 )
