from pathlib import Path
from setuptools import setup, find_packages


doc = Path(__file__).parent / 'README.md'


setup(name='tshistory',
      version='0.18.0',
      author='Pythonian',
      author_email='aurelien.campeas@pythonian.fr, arnaud.campeas@pythonian.fr',
      url='https://hg.sr.ht/~pythonian/tshistory',
      description='Timeseries store with version control',
      long_description=doc.read_text(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=[
          'pandas >= 1.0.5, < 1.5',
          'psyl',
          'dateutils',
          'sqlalchemy < 2',
          'sqlhelp',
          'click',
          'mock',
          'inireader',
          'pytz',
          'colorama',
          'tqdm',
          'flask < 2.2',
          'flask-restx >= 1.0.3, < 1.1',
          'jinja2 < 3.2',
          'requests'
      ],
      package_data={'tshistory': [
          'schema.sql',
          'series.sql',
          'group.sql'
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
