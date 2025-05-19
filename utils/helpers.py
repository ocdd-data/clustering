import json
import logging
import os
import time
from collections import defaultdict
from io import StringIO
from typing import Union

import pandas as pd
import requests


class Query:
  def __init__(self, id:int, params:dict=None):
    self.id = id
    self.params = params or {}

# Class definition to use Redash API
class Redash:
  def __init__(self, key:str, base_url:str) -> None:
    self.__API_KEY = key
    self.__BASE_URL = base_url
    self.job = defaultdict(lambda: None)
    self.resultId = defaultdict(lambda: None)
    # status -> 1: running, 2: completed, 3: failed
    self.status = defaultdict(lambda: None)

  def run_queries(self, queries:'list[Query]') -> None:
    for query in queries:
      self.run_query(query, batch=True)
    while queries:
      queries = list(filter(lambda query: self.status[query.id] == 1, queries))
      for query in queries:
        self.poll_job(query)
      time.sleep(1)
    
    # clear job dictonary when completed
    self.job = defaultdict(lambda: None)

  def run_query(self, query:Query, batch=False) -> None:
    payload = dict(max_age=0, parameters=query.params)

    res = requests.post(f'{self.__BASE_URL}/api/queries/{query.id}/results?api_key={self.__API_KEY}', data=json.dumps(payload), timeout=60)

    if res.status_code != 200:
      logging.warning(res.json())
      logging.warning(f'Refresh failed.')
      self.status[query.id] = 3
    else:
      self.status[query.id] = 1
      self.job[query.id] = res.json()['job']

    if not batch:
      while self.status[query.id] == 1:
        self.poll_job(query)
  
  def poll_job(self, query:Query) -> None:
    job = self.job[query.id]

    if job['status'] not in (3,4):
      response = requests.get(f"{self.__BASE_URL}/api/jobs/{job['id']}?api_key={self.__API_KEY}", timeout=60)
      self.job[query.id] = response.json()['job']

    elif job['status'] == 3:
      self.resultId[query.id] = job['query_result_id']
      self.status[query.id] = 2
      print(f'Query {query.id}: Completed.')

    elif job['status'] == 4:
      print(f'Query {query.id}: Execution failed.')
      self.status[query.id] = 3

  def read_csv_string(self, string:str) -> pd.DataFrame:
    # Convert string into StringIO
    csvStringIO = StringIO(string)

    # Load CSV string and return DataFrame
    return pd.read_csv(csvStringIO, sep=",")

  def get_result(self, query: Union[int,Query]) -> pd.DataFrame:
    queryId = query.id if type(query) is Query else query
    
    if queryId not in self.resultId:
      print(f'Query {queryId}: status {self.status[queryId]}')
    else:
      resultId = f'results/{self.resultId[queryId]}' if self.resultId[queryId] else 'results'
      res = requests.get(f'{self.__BASE_URL}/api/queries/{queryId}/{resultId}.csv?api_key={self.__API_KEY}', timeout=60)
      if res.status_code != 200:
        logging.warning(f'Failed getting results for Query {queryId}.')
      return self.read_csv_string(res.text)



