from setuptools import setup, find_packages
from typing import List

def get_requirements(filepath:str)->List[str]:
    HYPHEN_E_DOT = "-e ."
    requirements = []
    with open(filepath) as fileobj:
        requirements = fileobj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
    if HYPHEN_E_DOT in requirements:
        requirements.remove(HYPHEN_E_DOT)
    return requirements


setup(
    name="automl_app",
    version="0.0.1",
    author="susil",
    author_email="susilkumarkct@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)