import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.helpers import Query, Redash
from utils.slack import SlackBot
from utils.transition import generate_cluster_transition_barchart

class RiderClusterPredictor:
    def __init__(self, models_dir: Path):
        region = os.getenv("REGION")
        self.models_dir = models_dir / region

        scaler_def = pd.read_csv(self.models_dir / "scaler_params.csv")
        self.scaler_mean = scaler_def.set_index("Feature")["Mean"]
        self.scaler_var = scaler_def.set_index("Feature")["Variance"]

        cluster_def = pd.read_csv(self.models_dir / "cluster_centroids.csv")
        self.centroids = cluster_def["Centroid"].apply(eval).apply(np.array).tolist()
        self.centroids = np.vstack(self.centroids)

        self.label_map = dict(zip(cluster_def["Cluster"], cluster_def["Description"]))

    def assign(self, df):
        X = df[self.scaler_mean.index].dropna()
        X_scaled = (X - self.scaler_mean) / np.sqrt(self.scaler_var)
        dists = np.linalg.norm(X_scaled.values[:, np.newaxis] - self.centroids, axis=2)

        df = df.copy()
        df['cluster'] = np.argmin(dists, axis=1)
        df['cluster_name'] = df['cluster'].map(self.label_map)
        return df

def generate_cluster_summary_text(df_curr, df_prev, month_label, region):
    total_curr = len(df_curr)
    total_prev = len(df_prev)
    total_delta = total_curr - total_prev
    total_pct = (total_delta / total_prev * 100) if total_prev else 0
    arrow = ":increase:" if total_delta > 0 else ":decrease:" if total_delta < 0 else "➖"

    monthly_title = ":alphabet-white-m::alphabet-white-o::alphabet-white-n::alphabet-white-t::alphabet-white-h::alphabet-white-l::alphabet-white-y:"
    return (
        f"{monthly_title}\n*{region} Rider Segmentation Report ({month_label}):*\n"
        f"*Total Riders: {total_curr:,}* ({arrow} {abs(total_delta):,} | {total_pct:+.2f}%)\n"
    )

def build_slack_attachments(df_curr, df_prev, label_map):
    color_map = ["#FFB5A7", "#FFBE98", "#FCE5A1", "#A8E6C2", "#A7C7E7"]    
    attachments = []

    for i, cluster in enumerate(sorted(df_curr['cluster'].unique())):
        name = label_map[cluster]
        color = color_map[i % len(color_map)]

        curr_subset = df_curr[df_curr['cluster'] == cluster]
        prev_subset = df_prev[df_prev['cluster'] == cluster]

        curr_riders = len(curr_subset)
        prev_riders = len(prev_subset)
        delta_riders = curr_riders - prev_riders
        pct_riders = (delta_riders / prev_riders * 100) if prev_riders else 0

        curr_trips = curr_subset['count'].sum()
        prev_trips = prev_subset['count'].sum()
        delta_trips = curr_trips - prev_trips
        pct_trips = (delta_trips / prev_trips * 100) if prev_trips else 0

        avg_curr = curr_trips / curr_riders if curr_riders else 0
        avg_prev = prev_trips / prev_riders if prev_riders else 0
        delta_avg = avg_curr - avg_prev
        pct_avg = (delta_avg / avg_prev * 100) if avg_prev else 0

        def emoji(val): return ":increase:" if val > 0 else ":decrease:" if val < 0 else "➖"

        fields_block = {
            "title": f"*{name}*",
            "value": (
                f"*Total Riders:* {curr_riders:,} ({emoji(delta_riders)} {abs(delta_riders):,} | {pct_riders:+.2f}%)\n"
                f"*Total Trips:* {curr_trips:,} ({emoji(delta_trips)} {abs(delta_trips):,} | {pct_trips:+.2f}%)\n"
                f"*Avg Trips/Rider:* {avg_curr:.2f} ({emoji(delta_avg)} {abs(delta_avg):.2f} | {pct_avg:+.2f}%)"
            ),
            "short": False
        }

        attachments.append({
            "color": color,
            "fields": [fields_block]
        })

    return attachments

def main():
    load_dotenv()

    region = os.getenv("REGION")
    query_id = int(os.getenv("REPORT_ID"))

    today = datetime.today()
    first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    first_day_prev_month = (first_day_last_month - timedelta(days=1)).replace(day=1)

    output_month = first_day_last_month.strftime("%b %Y")
    prev_month_label = first_day_prev_month.strftime("%b %Y")
    redash_param_date = first_day_last_month.strftime("%Y-%m-%d")

    output_dir = f"output/{region}"
    os.makedirs(output_dir, exist_ok=True)

    client = Redash(os.getenv("REDASH_API_KEY"), os.getenv("REDASH_BASE_URL"))
    slack = SlackBot()

    pipeline = RiderClusterPredictor(models_dir=Path("models"))

    query = Query(query_id, params={"date": redash_param_date})
    client.run_queries([query])
    df_all = client.get_result(query.id)
    df_all['month'] = pd.to_datetime(df_all['month']).dt.date

    df_prev = df_all[df_all['month'] == first_day_prev_month.date()]
    df_curr = df_all[df_all['month'] == first_day_last_month.date()]

    df_prev_clustered = pipeline.assign(df_prev)
    df_curr_clustered = pipeline.assign(df_curr)

    summary_text = generate_cluster_summary_text(df_curr_clustered, df_prev_clustered, output_month, region)
    attachments = build_slack_attachments(df_curr_clustered, df_prev_clustered, pipeline.label_map)

    response = slack.client.chat_postMessage(
        channel=os.getenv("SLACK_CHANNEL"),
        text=summary_text,
        attachments=attachments
    )
    main_ts = response["ts"]

    df_prev_clustered["cluster"] = df_prev_clustered["cluster"].astype(str)
    df_curr_clustered["cluster"] = df_curr_clustered["cluster"].astype(str)
    chart_path, count_path, percent_path = generate_cluster_transition_barchart(
        df_prev_clustered, df_curr_clustered, prev_month_label, output_month, output_dir, region
    )

    slack.client.files_upload_v2(
        channel=os.getenv("SLACK_CHANNEL"),
        file=chart_path,
        initial_comment="📊 Cluster Transition Chart",
        thread_ts=main_ts
    )

    time.sleep(2)

    curr_path = f"{output_dir}/rider_clusters_{region}_{output_month}.csv"
    df_curr_clustered.to_csv(curr_path, index=False)

    slack.client.files_upload_v2(
        channel=os.getenv("SLACK_CHANNEL"),
        file=curr_path,
        initial_comment="📎 Rider Cluster Assignment CSV",
        thread_ts=main_ts
    )

if __name__ == "__main__":
    main()
