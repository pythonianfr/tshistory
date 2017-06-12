import click

from sqlalchemy import create_engine

from tshistory.tsio import TimeSerie


FMT = """
revision: {rev}
author:   {author}
date:     {date}
""".strip()

def format_rev(rev):
    fmt = FMT + '\n'
    if rev.get('diff'):
        fmt += 'series: {names}\n\n'
        lines = []
        for ts in rev['diff'].values():
            lines.append(ts.to_csv())
        fmt += '\n'.join(lines)
    else:
        fmt += 'series:   {names}'

    return fmt.format(**rev)


@click.command()
@click.argument('db-uri')
@click.option('--limit', '-l', default=None)
@click.option('--show-diff', is_flag=True, default=False)
@click.option('--serie', '-s', multiple=True)
def log(db_uri, limit, show_diff, serie):
    engine = create_engine(db_uri)

    tsh = TimeSerie()
    for rev in tsh.log(engine, limit=limit, diff=show_diff, names=serie):
        rev['names'] = ','.join(rev['names'])
        print(format_rev(rev))
        print()


if __name__ == '__main__':
    log()
