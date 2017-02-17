from pathlib import Path
from inireader.config import config
from sqlalchemy import create_engine

from tshistory.schema import meta


def init_db():
    here = Path(__file__).parent
    cfg = config(str(here / 'data_hub.cfg'))
    dburi = cfg['db']['uri']
    print(dburi)
    engine = create_engine(dburi)
    meta.drop_all(engine)
    meta.create_all(engine)


if __name__ == '__main__':
    response = input('Are you sure. This will erase the database (y/N)')
    if response.upper() == 'Y':
        print('creating the db schema ...')
        init_db()
