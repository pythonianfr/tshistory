from typing import Optional

from dateutil.relativedelta import relativedelta
import pandas as pd
import pytz
import tzlocal


OPERATORS = {}


def operator(name):

    def wrapper(func):
        OPERATORS[name] = func
        return func

    return wrapper


class horizon:
    __slots__ = 'past', 'future'

    def __init__(self, date, offset, past, future):
        d1 = date + past
        d2 = date + future
        delta = (d2 - d1) * offset
        self.past = d1 - delta
        self.future = d2 - delta


@operator('now')
def now(tz: Optional[str]='UTC') -> pd.Timestamp:
    if tz:
        pytz.timezone(tz)
    return pd.Timestamp.now(
        tz=tz or tzlocal.get_localzone()
    )


@operator('date')
def date(strdate: str,
         tz: Optional[str]='UTC') -> pd.Timestamp:
    if tz:
        pytz.timezone(tz)
    if tz is None:
        return pd.Timestamp(strdate)
    return pd.Timestamp(strdate, tz=tz)


@operator('delta')
def delta(years: int=0,
          months: int=0,
          weeks: int=0,
          days: int=0,
          hours: int=0) -> pd.Timestamp:
    return relativedelta(
        years=years,
        months=months,
        weeks=weeks,
        days=days,
        hours=hours
    )


@operator('horizon')
def _horizon(date: pd.Timestamp,
             offset: int,
             past: pd.Timedelta,
             future: pd.Timedelta) -> horizon:
    return horizon(date, offset, past, future)

