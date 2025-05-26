# === Imports
import os
from datetime import date, timedelta
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from utils.helpers import Redash, Query
from utils.slack import SlackBot
from models.clustering_week import scale_features, assign_clusters, get_redash_data

def main():
    # === Load environment variables
    load_dotenv()
    api_key = os.getenv("REDASH_API_KEY")
    base_url = os.getenv("REDASH_URL")
    query_id = 4641
    slack_channel = os.getenv("SLACK_CHANNEL")
    redash = Redash(key=api_key, base_url=base_url)

    # === Generate latest report for the most recent Monday
    today = date.today()
    report_monday = today if today.weekday() == 0 else today - timedelta(days=today.weekday())
    period_end = report_monday - timedelta(days=1)
    period_start = period_end - timedelta(weeks=4) + timedelta(days=1)

    print(f"\n📅 Generating report for: {report_monday} (covers {period_start} → {period_end})")
    df, cluster_stats = get_redash_data(start=period_start, end=period_end, redash=redash, query_id=query_id, return_cluster_stats=True)

    if df.empty:
        print("⚠️ No data found. Exiting.")
        return

    # === Count and trip data
    cluster_counts = df['Cluster_Description'].value_counts().to_dict()
    total_riders = len(df)
    report_key = report_monday.strftime("%Y-%m-%d")
    total_trips = df["total_trips"].sum()

    count_data = pd.Series(cluster_counts, name=report_key)
    count_data["Total"] = total_riders

    trip_data = cluster_stats.loc[report_key].copy()
    trip_data["Total"] = total_trips

    # === Average trips per rider by cluster (excluding Total)
    avg_trips = (trip_data.drop("Total") / count_data.drop("Total")).round(2)

    # === Build sections
    count_section = pd.DataFrame(count_data)
    count_section.index.name = "Weekly Cluster Counts"

    delta_section = pd.DataFrame("", index=[f"{label} (∆)" for label in count_data.index], columns=[report_key])
    delta_section.index.name = "Week-over-Week Changes"

    trip_section = pd.DataFrame(trip_data.drop("Total"))
    trip_section.columns = [report_key]
    trip_section.index.name = "Total Trips by Cluster"

    avg_section = pd.DataFrame(avg_trips)
    avg_section.columns = [report_key]
    avg_section.index.name = "Avg Trips per Rider by Cluster"

    blank_row = pd.DataFrame({report_key: [""]}, index=[""])

    final_report = pd.concat([
        count_section,
        blank_row,
        blank_row,
        delta_section,
        blank_row,
        blank_row,
        trip_section,
        blank_row,
        blank_row,
        avg_section
    ])

    # === Save and upload
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"weekly_rider_clusters_{report_key}.csv"
    output_path = os.path.join(output_dir, output_filename)
    final_report.to_csv(output_path)

    slack = SlackBot()
    if slack_channel:
        slack.uploadFile(output_path, slack_channel, f"Weekly Rider Segmentation Report for `{report_key}`")
        print(f"✅ Report uploaded to Slack: {slack_channel}")
    else:
        print("⚠️ SLACK_CHANNEL not set")

    print(f"✅ Report saved to {output_path}")

if __name__ == "__main__":
    main()
