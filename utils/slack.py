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
        filename = os.path.basename(file)
        filesize = os.path.getsize(file)

        try:
            # 1) Get upload URL
            ticket = self.client.files_getUploadURLExternal(
                filename=filename,
                length=filesize,
            )
            upload_url = ticket["upload_url"]
            file_id = ticket["file_id"]

            # 2) Upload raw bytes
            with open(file, "rb") as f:
                resp = requests.post(upload_url, files={"file": (filename, f)})
                resp.raise_for_status()

            # 3) Complete upload and share
            result = self.client.files_completeUploadExternal(
                files=[{"id": file_id, "title": filename}],
                channel_id=channel,
                initial_comment=comment if thread_ts is None else "",
                thread_ts=thread_ts,
            )
            self.logger.info(result)

            return result["files"][0]["timestamp"]

        except (SlackApiError, requests.RequestException) as e:
            self.logger.error(f"Error uploading file: {e}")
            return None

    def uploadFilesWithComment(self, files: list, channel: str, initial_comment: str = "", thread_ts: str = None) -> str:
        ts_to_return = thread_ts
        try:
            for idx, file_path in enumerate(files):
                filename = os.path.basename(file_path)
                filesize = os.path.getsize(file_path)

                # Step 1: Get upload URL
                ticket = self.client.files_getUploadURLExternal(
                    filename=filename,
                    length=filesize,
                )
                upload_url = ticket["upload_url"]
                file_id = ticket["file_id"]

                # Step 2: Upload raw bytes
                with open(file_path, "rb") as f:
                    resp = requests.post(upload_url, files={"file": (filename, f)})
                    resp.raise_for_status()

                # Step 3: Complete upload & share
                result = self.client.files_completeUploadExternal(
                    files=[{"id": file_id, "title": filename}],
                    channel_id=channel,
                    initial_comment=initial_comment if idx == 0 and ts_to_return is None else "",
                    thread_ts=ts_to_return,
                )

                # Capture thread timestamp from the first file
                if idx == 0 and ts_to_return is None:
                    ts_to_return = result["files"][0]["timestamp"]

            return ts_to_return

        except (SlackApiError, requests.RequestException) as e:
            self.logger.error(f"Error uploading files: {e}")
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
