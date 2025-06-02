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

    today = date.today()
    this_monday = today if today.weekday() == 0 else today - timedelta(days=today.weekday())
    last_monday = this_monday - timedelta(weeks=1)

    def get_bounds(monday):
        end = monday - timedelta(days=1)
        start = end - timedelta(weeks=4) + timedelta(days=1)
        return start, end

    data = {}
    for week in [last_monday, this_monday]:
        start, end = get_bounds(week)
        df, stats = get_redash_data(start=start, end=end, redash=redash, query_id=query_id, return_cluster_stats=True)
        if df.empty:
            print(f"No data for {week}")
            return

        cluster_counts = df['Cluster_Description'].value_counts().to_dict()
        cluster_counts["Total"] = len(df)
        stats_key = week.strftime("%Y-%m-%d")
        stats.loc[stats_key, "Total"] = df["total"].sum()
        avg_trips = (stats.loc[stats_key].drop("Total") / pd.Series(cluster_counts).drop("Total")).round(2)
        avg_trips["Total"] = stats.loc[stats_key, "Total"] / cluster_counts["Total"]

        data[stats_key] = {
            "count": pd.Series(cluster_counts),
            "trips": stats.loc[stats_key],
            "avg": avg_trips
        }

    prev, curr = sorted(data.keys())

    count_df = pd.DataFrame({prev: data[prev]["count"], curr: data[curr]["count"]})
    trip_df = pd.DataFrame({prev: data[prev]["trips"], curr: data[curr]["trips"]})
    avg_df = pd.DataFrame({prev: data[prev]["avg"], curr: data[curr]["avg"]})

    count_delta = (count_df[curr] - count_df[prev]).astype(int)
    count_pct = ((count_delta / count_df[prev]) * 100).round(2)

    trip_delta = (trip_df[curr] - trip_df[prev]).round()
    trip_pct = ((trip_delta / trip_df[prev]) * 100).round(2)

    avg_delta = (avg_df[curr] - avg_df[prev]).round(2)
    avg_pct = ((avg_delta / avg_df[prev]) * 100).round(2)

    movement_impact = (count_delta.drop("Total") * avg_df[curr].drop("Total")).round()
    movement_impact["Total"] = movement_impact.sum()

    # === Format CSV Report (only show "Total" for counts and movement impact)
    records = []

    # Section 1: Weekly Cluster Counts (include Total)
    records.append(["Weekly Cluster Counts (Rolling 4-Week View)", ""])
    for cluster in count_df.index:
        records.append([cluster, count_df[curr][cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # Section 2: Weekly Cluster Counts Δ (include Total)
    records.append(["Weekly Cluster Counts Δ by Cluster", ""])
    for cluster in count_df.index:
        records.append([f"{cluster} (Δ)", count_delta[cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # Section 3: Total Trips by Cluster (exclude Total)
    records.append(["Total Trips by Cluster", ""])
    for cluster in trip_df.index.drop("Total"):
        records.append([cluster, trip_df[curr][cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # Section 4: Total Trips Δ by Cluster (exclude Total)
    records.append(["Total Trips Δ by Cluster", ""])
    for cluster in trip_df.index.drop("Total"):
        records.append([f"{cluster} (Δ)", trip_delta[cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # Section 5: Avg Total Trips per Rider (exclude Total)
    records.append(["Avg Total Trips per Rider", ""])
    for cluster in avg_df.index.drop("Total"):
        records.append([cluster, avg_df[curr][cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # Section 6: Avg Total Trips Δ by Cluster (exclude Total)
    records.append(["Avg Total Trips Δ by Cluster", ""])
    for cluster in avg_df.index.drop("Total"):
        records.append([f"{cluster} (Δ)", avg_delta[cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # Section 7: Estimated Movement Impact by Cluster (include Total)
    records.append(["Estimated Movement Impact by Cluster", ""])
    for cluster in movement_impact.index:
        records.append([cluster, movement_impact[cluster]])
    records.append(["", ""])
    records.append(["", ""])

    # === Save to CSV
    final_df = pd.DataFrame(records, columns=["Metric", curr])
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"weekly_rider_clusters_{curr}.csv")
    final_df.to_csv(output_path, index=False)

    # === Build Slack attachments
    color_map = ["#ffb5a7", "#ffbe98", "#fce5a1", "#a8e6c2", "#a7c7e7"]
    slack = SlackBot()

    attachments = []
    for i, cluster in enumerate(count_df.index.drop("Total")):
        color = color_map[i % len(color_map)]

        def get_emoji(val):
            return ":increase:" if val > 0 else ":decrease:" if val < 0 else ":heavy_minus_sign:"

        emoji_count = get_emoji(count_delta[cluster])
        emoji_trip = get_emoji(trip_delta[cluster])
        emoji_avg = get_emoji(avg_delta[cluster])

        field = {
            "title": cluster,
            "value": (
                f"*Rider Count:* {count_df[curr][cluster]} "
                f"({count_delta[cluster]:+} | {count_pct[cluster]:+}%) {emoji_count}\n"
                f"*Total Trips:* {trip_df[curr][cluster]} "
                f"({trip_delta[cluster]:+} | {trip_pct[cluster]:+}%) {emoji_trip}\n"
                f"*Average Trips:* {avg_df[curr][cluster]:.2f} "
                f"({avg_delta[cluster]:+0.2f} | {avg_pct[cluster]:+0.2f}%) {emoji_avg}"
            ),
            "short": False
        }

        attachments.append({
            "color": color,
            "fields": [field]
        })


    response = slack.client.chat_postMessage(
        channel=slack_channel,
        text=(
            "*:alphabet-white-w::alphabet-white-e::alphabet-white-e:"
            ":alphabet-white-k::alphabet-white-l::alphabet-white-y:*\n"
            "*Rider Segmentation Report*\n"
            f"`{curr}` (covers {period_start} → {period_end})\n"
            f"vs\n"
            f"`{prev}` (covers {prev_start} → {prev_end})"
        ),
        attachments=attachments
    )

    slack.client.files_upload_v2(
        channel=slack_channel,
        file=output_path,
        initial_comment="📎 Attached CSV report:",
        thread_ts=response["ts"]
    )


if __name__ == "__main__":
    main()
