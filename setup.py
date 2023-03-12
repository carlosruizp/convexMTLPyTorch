from setuptools import setup
from setuptools import find_namespace_packages


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='convexmtl-torch',
      version='0.1',
      description='Package for Multiple Task Learning using sklearn guidelines',
      long_description=readme(),
      url='http://github.com/storborg/funniest',
      author='Carlos Ruiz Pastor',
      author_email='carlosruizpastor@protonmail.com',
      license='MIT',
      # packages=['convexmtl_torch.model', 'convexmtl_torch.data'],
      packages=find_namespace_packages(include=['convexmtl_torch.*']),
      install_requires=[
          'torch', 'numpy', 'pytorch-lightning'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],
      #scripts=['bin/funniest-joke'],
    )
