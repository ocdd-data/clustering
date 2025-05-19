import os
from io import StringIO
from logging import Logger

import pandas as pd
import requests
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


class SlackBot:
  def __init__(self):
    load_dotenv()

    self.client = WebClient(token=os.getenv('SLACK_TOKEN'))
    self.logger = Logger('SlackBot')

  def uploadFile(self, file:str, channel:str, comment:str) -> None:
    try:
      result = self.client.files_upload_v2(
        channel=channel,
        initial_comment=comment,
        file=file,
      )
      self.logger.info(result)
    except SlackApiError as e:
      try:
        self.client.chat_postMessage(
          channel=channel,
          text=f"Error uploading file: {e}"
        )
      except SlackApiError as e:
        self.logger.error("Error uploading file: {}".format(e))

  def to_pandas(self, url:str) -> pd.DataFrame:
    response = requests.get(url, headers={'Authorization': f'Bearer {os.getenv("SLACK_TOKEN")}'}, timeout=60)

    return pd.read_csv(StringIO(response.text))
    
  def getLatestFile(self, channel:str) -> pd.DataFrame:
    try:
      response = self.client.files_list(
        channel=channel,
        limit=1,
        latest=True
      )
      file = response['files'][0]
      file_url_private = file['url_private']
      return self.to_pandas(file_url_private)
    except KeyError:
      self.logger.error("No files found in channel")
      return None
    except SlackApiError as e:
      self.logger.error(f"Error retrieving latest file message: {e}")
      return None

  def deleteLatestMessage(self, channel:str) -> None:
    try:
      response = self.client.conversations_history(
        channel=channel,
        limit=1
      )
      message = response['messages'][0]
      self.client.chat_delete(
        channel=channel,
        ts=message['ts']
      )
    except SlackApiError as e:
      self.logger.error(f"Error deleting latest file message: {e}")
