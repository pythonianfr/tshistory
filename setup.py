from setuptools import setup


setup(name='tshistory',
      version='0.1',
      author='Pythonian',
      author_email='aurelien.campeas@pythonian.fr, arnaud.campeas@pythonian.fr',
      description='Store timeseries histories into postgres',

      packages=['tshistory'],
      package_dir={'tshistory': '.'},
      install_requires=[
          'pathlib',
          'pandas',
          'sqlalchemy',
          'click',
          'click-plugins',
          'mock',
          'pytest_sa_pg',
      ],
      entry_points={
          'console_scripts': [
              'tsh=tshistory.command:tsh'
          ]}
)
