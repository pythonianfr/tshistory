from setuptools import setup


setup(name='tshistory',
      version='0.1',
      author='Pythonian',
      author_email='aurelien.campeas@pythonian.fr, arnaud.campeas@pythonian.fr',
      description='Store timeseries histories into postgres',

      packages=['tshistory'],
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
          ]},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
          'Topic :: Database',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development :: Version Control'
      ]
)
