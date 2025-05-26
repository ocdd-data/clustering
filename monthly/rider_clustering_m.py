import os
import ast
import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timedelta
from dotenv import load_dotenv
from utils.helpers import Query, Redash
from utils.slack import SlackBot
from utils.transition import generate_cluster_transition_barchart

class RiderClusterPredictor:
    def __init__(self, models_dir: Path):
        scaler_def = pd.read_csv(models_dir / "scaler_params.csv")
        self.scaler_mean = scaler_def.set_index("Feature")["Mean"]
        self.scaler_var = scaler_def.set_index("Feature")["Variance"]

        cluster_def = pd.read_csv(models_dir / "cluster_centroids.csv")
        self.centroids = cluster_def[self.scaler_mean.index.tolist()].values
        self.label_map = dict(zip(cluster_def["Cluster"], cluster_def["Description"]))

    def assign(self, df):
        X = df[self.scaler_mean.index].dropna()
        X_scaled = (X - self.scaler_mean) / np.sqrt(self.scaler_var)
        dists = np.linalg.norm(X_scaled.values[:, np.newaxis] - self.centroids, axis=2)
        df = df.copy()
        df['cluster'] = np.argmin(dists, axis=1)
        df['cluster_name'] = df['cluster'].map(self.label_map)
        return df

def generate_cluster_summary_with_diff(df_curr, df_prev, label_map, month_label, prev_label, region):
    curr_counts = df_curr['cluster'].value_counts().rename("curr_count")
    prev_counts = df_prev['cluster'].value_counts().rename("prev_count")
    diff_df = pd.concat([curr_counts, prev_counts], axis=1).fillna(0).astype(int)
    diff_df["delta"] = diff_df["curr_count"] - diff_df["prev_count"]
    diff_df["cluster_name"] = diff_df.index.map(label_map)

    lines = [f"*{region} Rider Segmentation Report ({month_label}):*"]
    for idx, row in diff_df.iterrows():
        delta = row["delta"]
        arrow = ":increase:" if delta > 0 else ":decrease:" if delta < 0 else "➖"
        delta_str = f"{arrow} {abs(delta):,}"
        lines.append(f"> *{row['cluster_name']}*: *{row['curr_count']:,}* ({delta_str})")

    return "\n".join(lines)

def main():
    load_dotenv()

    region = "SG"
    today = datetime.today()
    first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    first_day_prev_month = (first_day_last_month - timedelta(days=1)).replace(day=1)

    output_month = first_day_last_month.strftime("%b_%Y")
    prev_month_label = first_day_prev_month.strftime("%b_%Y")
    redash_param_date = first_day_last_month.strftime("%Y-%m-%d")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    client = Redash(os.getenv("REDASH_API_KEY"), os.getenv("REDASH_BASE_URL"))
    slack = SlackBot()

    pipeline = RiderClusterPredictor(models_dir=Path("models"))

    query = Query(4216, params={"date": redash_param_date})
    client.run_queries([query])
    df_all = client.get_result(query.id)
    df_all['month'] = pd.to_datetime(df_all['month']).dt.date

    df_prev = df_all[df_all['month'] == first_day_prev_month.date()]
    df_curr = df_all[df_all['month'] == first_day_last_month.date()]

    df_prev_clustered = pipeline.assign(df_prev)
    df_curr_clustered = pipeline.assign(df_curr)

    summary = generate_cluster_summary_with_diff(
        df_curr_clustered, df_prev_clustered, pipeline.label_map,
        output_month, prev_month_label, region
    )
    summary_ts = slack.postMessage(os.getenv("SLACK_CHANNEL"), summary)

    curr_path = f"{output_dir}/rider_clusters_{region}_{output_month}.csv"
    df_curr_clustered.to_csv(curr_path, index=False)

    slack.uploadFile(curr_path, os.getenv("SLACK_CHANNEL"), comment="Cluster assignments CSV", thread_ts=summary_ts)

    generate_cluster_transition_barchart(
        df_prev_clustered,
        df_curr_clustered,
        prev_month_label,
        output_month,
        output_dir,
        slack,
        thread_ts=None
    )

if __name__ == "__main__":
    main()
