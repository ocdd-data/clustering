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

    def summarize(self, df):
        summary = df['cluster_name'].value_counts().rename_axis('cluster_name').reset_index(name='count')
        return summary.to_string(index=False)

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

    df_prev_clustered.to_csv(f"{output_dir}/rider_clusters_{region}_{prev_month_label}.csv", index=False)
    df_prev_clustered.to_csv(f"{output_dir}/rider_clusters_{region}_prev.csv", index=False)

    curr_path = f"{output_dir}/rider_clusters_{region}_{output_month}.csv"
    df_curr_clustered.to_csv(curr_path, index=False)

    summary = pipeline.summarize(df_curr_clustered)
    slack.uploadFile(curr_path, os.getenv("SLACK_CHANNEL"), f"*Rider Segmentation* for {region} ({output_month})\n```{summary}```")

    generate_cluster_transition_barchart(
      df_prev_clustered,
      df_curr_clustered,
      prev_month_label,
      output_month,
      output_dir,
      slack
    )

if __name__ == "__main__":
    main()
