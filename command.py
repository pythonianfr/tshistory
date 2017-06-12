import click

from sqlalchemy import create_engine

from tshistory.tsio import TimeSerie


REVFMT = """
revision: {rev}
author:   {author}
date:     {date}
""".strip()

def format_rev(rev):
    fmt = REVFMT + '\n'
    if rev.get('diff'):
        fmt += 'series: {names}\n\n'
        lines = []
        for ts in rev['diff'].values():
            lines.append(ts.to_csv())
        fmt += '\n'.join(lines)
    else:
        fmt += 'series:   {names}'

    return fmt.format(**rev)


@click.group()
def tsh():
    pass


@tsh.command()
@click.argument('db-uri')
@click.option('--limit', '-l', default=None)
@click.option('--show-diff', is_flag=True, default=False)
@click.option('--serie', '-s', multiple=True)
@click.option('--from-rev')
@click.option('--to-rev')
def log(db_uri, limit, show_diff, serie, from_rev, to_rev):
    engine = create_engine(db_uri)

    tsh = TimeSerie()
    for rev in tsh.log(engine, limit=limit, diff=show_diff, names=serie,
                       fromrev=from_rev, torev=to_rev):
        rev['names'] = ','.join(rev['names'])
        print(format_rev(rev))
        print()


INFOFMT = """
changeset count: {changeset count}
series count:    {series count}
series names:    {serie names}
""".strip()

@tsh.command()
@click.argument('db-uri')
def info(db_uri):
    engine = create_engine(db_uri)

    tsh = TimeSerie()
    info = tsh.info(engine)
    info['serie names'] = ', '.join(info['serie names'])
    print(INFOFMT.format(**info))


if __name__ == '__main__':
    tsh()
