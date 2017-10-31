import json
from zipfile import ZipFile
from datetime import datetime

import pandas as pd
from sqlalchemy import create_engine

from tshistory.tsio import tojson
from tshistory.helper import tempdir


def dump(dburi, dump_path, tsh, additional_dumping):
    engine = create_engine(dburi)

    with engine.connect() as cn:
        logs = tsh.log(cn)

    with tempdir() as temp_dir:
        # this injected callback is an extension point for
        # components that specialize tshistory
        additional_dumping(dburi, temp_dir)

        pd.DataFrame(logs).to_csv(str(temp_dir / 'registry.csv'))
        for cset in logs:
            csid = cset['rev']
            with engine.connect() as cn:
                cset_diff = tsh.log(cn, diff=True, fromrev=csid, torev=csid)[0]

            for name in cset['names']:
                cset_diff['diff'][name] = tojson(cset_diff['diff'][name])
                cset_diff['date'] = str(cset_diff['date'])

            (temp_dir / str(csid)).write_bytes(json.dumps(cset_diff).encode('utf-8'))
            print(str(csid) + ' / ' + str(len(logs)))

        out_path = str(dump_path / ('dump_{}.zip'.format(
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
        )
        with ZipFile(out_path, 'w') as myzip:
            for file in temp_dir.iterdir():
                myzip.write(str(file), file.name)
    return out_path


def restore(out_path, dburi, tsh, read_and_insert, additional_restoring):
    engine = create_engine(dburi)
    logs = tsh.log(engine, limit=1)
    if logs:
        print("I'm afraid I can't to this, Dave. The new database is not empty.")
        return

    with ZipFile(out_path, 'r') as myzip:
        additional_restoring(out_path, dburi)
        df_registry = pd.read_csv(myzip.open('registry.csv'))
        maxrevs = max(df_registry['rev'])
        for csid in sorted(df_registry['rev']):
            with myzip.open(str(csid)) as cset_file:
                cset_json = cset_file.read().decode('utf-8')

            with engine.connect() as cn:
                read_and_insert(cn, tsh, cset_json)

            print(str(csid) + ' / ' + str(maxrevs))
