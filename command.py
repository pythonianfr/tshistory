from pkg_resources import iter_entry_points

import click
from click_plugins import with_plugins

from sqlalchemy import create_engine

from tshistory.tsio import TimeSerie


TSH = TimeSerie()


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
            lines.append(ts.to_string())
        fmt += '\n'.join(lines)
    else:
        fmt += 'series:   {names}'

    return fmt.format(**rev)


@with_plugins(iter_entry_points('tshistory.subcommands'))
@click.group(invoke_without_command=True)
@click.pass_context
def tsh(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_usage())


@tsh.command()
@click.argument('db-uri')
@click.option('--limit', '-l', default=None)
@click.option('--show-diff', is_flag=True, default=False)
@click.option('--serie', '-s', multiple=True)
@click.option('--from-rev')
@click.option('--to-rev')
def log(db_uri, limit, show_diff, serie, from_rev, to_rev):
    engine = create_engine(db_uri)

    for rev in TSH.log(engine, limit=limit, diff=show_diff, names=serie,
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

    info = TSH.info(engine)
    info['serie names'] = ', '.join(info['serie names'])
    print(INFOFMT.format(**info))


if __name__ == '__main__':
    tsh()
