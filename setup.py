from setuptools import setup, find_packages
from os.path import join, dirname, abspath
import readytofit

HERE = abspath(dirname(__file__))
with open(join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

setup(
    name='ready-to-fit',
    author='Fiodar Drazdou',
    author_email='dfgckpsk@gmail.com',
    version=readytofit.__version__,
    include_package_data=True,
    package_data={'project': ['readytofit/tools/log_cfg.ini', 'tools/log_cfg.ini']},
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    install_requires=install_reqs,
    python_requires='>=3.6',
    test_suite='readytofit.tests'
)