from setuptools import setup


setup(name='tshistory',
      version='0.7.0',
      author='Pythonian',
      author_email='aurelien.campeas@pythonian.fr, arnaud.campeas@pythonian.fr',
      url='https://bitbucket.org/pythonian/tshistory',
      description='Store timeseries histories into postgres',

      packages=['tshistory'],
      install_requires=[
          'pandas ~= 0.24',
          'dateutils',
          'sqlalchemy',
          'sqlhelp',
          'click',
          'mock',
          'pytest_sa_pg',
          'inireader',
          'colorama',
          'tqdm'
      ],
      entry_points={
          'console_scripts': [
              'tsh=tshistory.cli:tsh'
          ]},
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 3',
          'Topic :: Database',
          'Topic :: Scientific/Engineering',
          'Topic :: Software Development :: Version Control'
      ]
)
