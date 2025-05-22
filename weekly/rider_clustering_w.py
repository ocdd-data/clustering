# === Imports
import os
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import re
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

from utils.helpers import Redash, Query
from utils.slack import SlackBot
from models.clustering import scale_features, assign_clusters, get_redash_data

# === Load environment variables
load_dotenv()

# === Redash setup
api_key = os.getenv("REDASH_API_KEY")
base_url = os.getenv("REDASH_URL")
query_id = 4641
redash = Redash(key=api_key, base_url=base_url)

# === Generate latest report for the most recent Monday
today = date.today()
report_monday = today if today.weekday() == 0 else today - timedelta(days=today.weekday())
report_mondays = [report_monday]

weekly_counts = defaultdict(dict)
weekly_total_trips = defaultdict(dict)

for report_monday in report_mondays:
    period_end = report_monday - timedelta(days=1)
    period_start = period_end - timedelta(weeks=4) + timedelta(days=1)

    print(f"Report for: {report_monday} (covers {period_start} → {period_end})")

    df, cluster_stats = get_redash_data(period_start, period_end, redash, query_id, return_cluster_stats=True)
    if df.empty:
        continue

    cluster_counts = df['Cluster_Description'].value_counts().to_dict()
    for label, count in cluster_counts.items():
        weekly_counts[label][report_monday] = count
    weekly_counts['Total'][report_monday] = len(df)

    report_key = report_monday.strftime("%Y-%m-%d")
    for label in cluster_stats.columns:
        weekly_total_trips[label][report_monday] = cluster_stats.at[report_key, label]
    weekly_total_trips['Total'][report_monday] = df['total_trips'].sum()

# === Build report sections
report_date = report_mondays[0]
report_date_str = report_date.strftime("%Y-%m-%d")

count_df = pd.DataFrame(weekly_counts).fillna(0).T.sort_index()
delta_df = count_df.diff(axis=1)
trips_df = pd.DataFrame(weekly_total_trips).fillna(0).T.sort_index()

section1 = pd.DataFrame(count_df.loc[report_date].values, columns=[report_date_str])
section2 = pd.DataFrame("", index=range(len(count_df.columns)), columns=[report_date_str])
section3 = pd.DataFrame(trips_df.loc[report_date].values, columns=[report_date_str])

blank_row = pd.DataFrame({report_date_str: [""]})

final_report = pd.concat([
    section1,
    blank_row,
    blank_row,
    section2,
    blank_row,
    blank_row,
    section3
], ignore_index=True)

# === Save and upload
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

output_filename = f"weekly_rider_clusters_{report_date_str}.csv"
output_path = os.path.join(output_dir, output_filename)
final_report.to_csv(output_path, index=False)

slack = SlackBot()
slack.uploadFile(
    output_path,
    os.getenv("SLACK_CHANNEL"),
    f"Weekly Rider Segmentation Report for `{report_date_str}`"
)

print(f"Combined report saved to {output_path}")
