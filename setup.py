from pathlib import Path
from setuptools import setup, find_packages

from tshistory import __version__


doc = Path(__file__).parent / 'README.md'


setup(name='tshistory',
      version=__version__,
      author='Pythonian',
      author_email='aurelien.campeas@pythonian.fr, arnaud.campeas@pythonian.fr',
      url='https://hg.sr.ht/~pythonian/tshistory',
      description='Timeseries store with version control',
      long_description=doc.read_text(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      python_requires='>= 3.9',
      install_requires=[
          'pandas >= 1.0.5, < 1.6',
          'dbcache >= 0.4.0',
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
          'flask == 2.3.2',
          'flask-restx == 1.1',
          'jinja2 < 3.2',
          'requests',
          'version-parser == 1.0.1',
          'importlib-metadata == 6.8'
      ],
      package_data={'tshistory': [
          'schema.sql',
          'series.sql',
          'group.sql'
      ]},
      entry_points={
          'console_scripts': [
              'tsh=tshistory.cli:tsh'
          ],
          'tshistory.migrate.Migrator': [
              'migrator=tshistory.migrate:Migrator'
          ],
          'tshclass': [
              'tshclass=tshistory.tsio:timeseries'
          ],
          'httpclient': [
              'httpclient=tshistory.http.client:httpclient'
          ],
          'forceimports': [
              # empty there, but plugins will add items to this family
              # to make sure timely objects registration happens
          ]
      },
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
