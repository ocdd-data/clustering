# === Imports
import os
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
import pandas as pd
from dotenv import load_dotenv
from collections import defaultdict
from utils.helpers import Redash, Query
from utils.slack import SlackBot
from models.clustering_weekly import get_redash_data

def main():
    load_dotenv()
    api_key = os.getenv("REDASH_API_KEY")
    base_url = os.getenv("REDASH_BASE_URL")
    query_id = 4641
    slack_channel = os.getenv("SLACK_CHANNEL")
    redash = Redash(key=api_key, base_url=base_url)

    # Get most recent Monday
    today = date.today()
    report_monday = today if today.weekday() == 0 else today - timedelta(days=today.weekday())
    period_end = report_monday - timedelta(days=1)
    period_start = period_end - timedelta(weeks=4) + timedelta(days=1)
    report_key = report_monday.strftime("%Y-%m-%d")

    print(f"\n📅 Generating report for: {report_monday} (covers {period_start} → {period_end})")
    df, cluster_stats = get_redash_data(start=period_start, end=period_end, redash=redash, query_id=query_id, return_cluster_stats=True)

    if df.empty:
        print("⚠️ No data found. Skipping.")
        return

    # Count per cluster
    cluster_counts = df['Cluster_Description'].value_counts().to_dict()
    total_riders = len(df)
    cluster_counts["Total"] = total_riders
    count_series = pd.Series(cluster_counts, name=report_key)

    # Trips per cluster
    trip_series = cluster_stats.loc[report_key]
    trip_series["Total"] = df["total"].sum()

    # Average trips per rider by cluster (exclude Total)
    avg_series = (trip_series.drop("Total") / count_series.drop("Total")).round(2)

    # Build sections
    count_df = pd.DataFrame(count_series)
    count_df.index.name = "Weekly Cluster Counts"

    delta_df = pd.DataFrame("", index=[f"{label} (∆)" for label in count_df.index], columns=[report_key])
    delta_df.index.name = "Week-over-Week Changes"

    trip_df = pd.DataFrame(trip_series.drop("Total"))
    trip_df.columns = [report_key]
    trip_df.index.name = "Total Trips by Cluster"

    avg_df = pd.DataFrame(avg_series)
    avg_df.columns = [report_key]
    avg_df.index.name = "Avg Trips per Rider by Cluster"

    blank_row = pd.DataFrame({report_key: [""]}, index=[""])

    final_df = pd.concat([
        count_df,
        blank_row,
        blank_row,
        delta_df,
        blank_row,
        blank_row,
        trip_df,
        blank_row,
        blank_row,
        avg_df
    ])

    # Save
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/weekly_rider_clusters_{report_key}.csv"
    final_df.to_csv(output_path)

    slack = SlackBot()
    slack.postMessage(
        channel=slack_channel,
        text=f"📊 Weekly Rider Segmentation Report for `{report_key}`"
    )

        # Upload file (no initial comment = no preview)
    slack.uploadFile(
        file=output_path,
        channel=slack_channel,
        comment="",  # prevents preview from showing
    )


if __name__ == "__main__":
    main()
