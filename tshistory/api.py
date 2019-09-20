from datetime import datetime
from typing import (
    Dict,
    Optional,
    Union
)
from sqlalchemy import create_engine
import pandas as pd

from tshistory.tsio import timeseries as dbtimeseries


class timeseries:
    __slots__ = ('engine', 'tsh')

    def __init__(self,
                 uri: str,
                 namespace: str='tsh'):
        self.engine = create_engine(uri)
        self.tsh = dbtimeseries(namespace)

    def update(self,
               updatets: pd.Series,
               name: str,
               author: str,
               metadata: Optional[dict]=None,
               insertion_date: Optional[datetime]=None) -> Optional[pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.update(
                cn,
                updatets,
                name,
                author,
                metadata=metadata,
                insertion_date=insertion_date
            )

    def replace(self,
                updatets: pd.Series,
                name: str,
                author: str,
                metadata: Optional[dict]=None,
                insertion_date: Optional[datetime]=None) -> Optional[pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.replace(
                cn,
                updatets,
                name,
                author,
                metadata=metadata,
                insertion_date=insertion_date
            )

    def get(self, name: str,
            revision_date: Optional[datetime]=None,
            from_value_date: Optional[datetime]=None,
            to_value_date: Optional[datetime]=None) -> Optional[pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.get(
                cn,
                name,
                revision_date=revision_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date
            )

    def history(self,
                name: str,
                from_insertion_date: Optional[datetime]=None,
                to_insertion_date: Optional[datetime]=None,
                from_value_date: Optional[datetime]=None,
                to_value_date: Optional[datetime]=None,
                diffmode: bool=False) -> Dict[datetime, pd.Series]:
        with self.engine.begin() as cn:
            return self.tsh.history(
                cn,
                name,
                from_insertion_date=from_insertion_date,
                to_insertion_date=to_insertion_date,
                from_value_date=from_value_date,
                to_value_date=to_value_date,
                diffmode=diffmode
            )

