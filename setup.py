from pathlib import Path
from setuptools import setup


doc = Path(__file__).parent / 'README.md'


setup(name='tshistory',
      version='0.9.0',
      author='Pythonian',
      author_email='aurelien.campeas@pythonian.fr, arnaud.campeas@pythonian.fr',
      url='https://bitbucket.org/pythonian/tshistory',
      description='Timeseries store with version control',
      long_description=doc.read_text(),
      long_description_content_type='text/markdown',
      packages=['tshistory'],
      install_requires=[
          'pandas >= 0.24, < 0.26',
          'dateutils',
          'sqlalchemy',
          'sqlhelp',
          'click',
          'mock',
          'inireader',
          'colorama',
          'tqdm',
          'deprecated'
      ],
      package_data={'tshistory': [
          'schema.sql',
          'series.sql'
      ]},
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
