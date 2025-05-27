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
        self.channel = os.getenv("SLACK_CHANNEL")

    def postMessage(self, channel: str, text: str) -> str:
        try:
            response = self.client.chat_postMessage(channel=channel, text=text)
            self.logger.info(response)
            return response["ts"]  
        except SlackApiError as e:
            self.logger.error(f"Error posting message: {e}")
            return None

    def uploadFile(self, file: str, channel: str, comment: str = "", thread_ts: str = None) -> str:
        try:
            response = self.client.files_upload_v2(
                channel=channel,
                file=file,
                initial_comment=comment if not thread_ts else None,
                thread_ts=thread_ts
            )
            self.logger.info(response)

            return response["file"]["timestamp"]

        except SlackApiError as e:
            self.logger.error(f"Error uploading file: {e}")
            return None


    def uploadFilesWithComment(self, files: list, channel: str, initial_comment: str = "") -> str:
        ts_to_return = None
        for idx, file_path in enumerate(files):
            try:
                result = self.client.files_upload_v2(
                    channel=channel,
                    file=file_path,
                    initial_comment=initial_comment if idx == 0 else None
                )
                self.logger.info(result)

                if ts_to_return is None:
                    shares = result.get("file", {}).get("shares", {})
                    for scope in ["public", "private"]:
                        if scope in shares and channel in shares[scope]:
                            ts_to_return = shares[scope][channel][0]["ts"]

            except SlackApiError as e:
                self.logger.error(f"Error uploading file {file_path}: {e}")
        return ts_to_return


    def to_pandas(self, url: str) -> pd.DataFrame:
        response = requests.get(url, headers={'Authorization': f'Bearer {os.getenv("SLACK_TOKEN")}'}, timeout=60)
        return pd.read_csv(StringIO(response.text))

    def getLatestFile(self, channel: str) -> pd.DataFrame:
        try:
            response = self.client.files_list(channel=channel, limit=1)
            file = response['files'][0]
            file_url_private = file['url_private']
            return self.to_pandas(file_url_private)
        except (KeyError, SlackApiError) as e:
            self.logger.error(f"Error retrieving latest file: {e}")
            return None

    def deleteLatestMessage(self, channel: str) -> None:
        try:
            response = self.client.conversations_history(channel=channel, limit=1)
            message = response['messages'][0]
            self.client.chat_delete(channel=channel, ts=message['ts'])
        except SlackApiError as e:
            self.logger.error(f"Error deleting latest message: {e}")
