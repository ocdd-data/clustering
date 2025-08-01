import os
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from utils.helpers import Query, Redash
from utils.slack import SlackBot


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
        df = df.loc[X.index]
        df['cluster'] = np.argmin(dists, axis=1)
        df['cluster_name'] = df['cluster'].map(self.label_map)
        return df


def generate_slack_text_summary(df_clustered, week_ending_str):
    counts = df_clustered.groupby("cluster_name").size().sort_index()

    header = f"📊 Rider Clustering — Week Ending {week_ending_str}\n"
    lines = [
        f"{cluster_name}: {count:,}"
        for cluster_name, count in counts.items()
    ]
    return header + "\n" + "\n".join(lines)


def run_weekly_clustering(reference_monday: datetime, custom_end_date: datetime = None):
    region = os.getenv("REGION")
    query_id = int(os.getenv("REPORT_ID"))
    models_dir = Path("models")

    start_date = reference_monday.replace(day=1)
    end_date = custom_end_date if custom_end_date else reference_monday - timedelta(days=1)  # Sunday before current Monday
    
    # Only skip if it's a weekly run (no custom_end_date provided)
    if start_date == reference_monday and custom_end_date is None:
        print(f"Skipping {reference_monday.strftime('%Y-%m-%d')} (1st of month is Monday, no weekly data yet)")
        return None

    redash = Redash(os.getenv("REDASH_API_KEY"), os.getenv("REDASH_BASE_URL"))
    slack = SlackBot()

    date_range_start = start_date.strftime("%Y-%m-%d")
    date_range_end = end_date.strftime("%Y-%m-%d")

    query = Query(query_id, params={
        "start_date": date_range_start,
        "end_date": date_range_end
    })

    redash.run_queries([query])
    df = redash.get_result(query.id)

    predictor = RiderClusterPredictor(models_dir=models_dir)

    df_clustered = predictor.assign(df)

    # Prepare summary counts with Sunday date as column name
    column_name = end_date.strftime('%d/%m/%y')
    df_summary = df_clustered.groupby("cluster_name").size().reset_index(name=column_name)

    all_labels = list(predictor.label_map.values())
    df_summary = df_summary.set_index("cluster_name").reindex(all_labels, fill_value=0).reset_index()

    slack_text = "cluster_name," + column_name + "\n" + "\n".join([
        f"{row['cluster_name']},{row[column_name]}" for _, row in df_summary.iterrows()
    ])

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    output_filename = output_dir / f"EW_Weekly_Rider_Clustering_{start_date.strftime('%-d')}-{end_date.strftime('%-d%b').lower()}.csv"
    df_summary.to_csv(output_filename, index=False)

    slack.client.files_upload(
        channels=os.getenv("SLACK_CHANNEL"),
        file=str(output_filename),
        title=f"Weekly Expanding Clustering Summary — {start_date.strftime('%-d')}–{end_date.strftime('%-d %B %Y')}",
        initial_comment=f":chart_with_upwards_trend: *Weekly Expanding Clustering Summary: {start_date.strftime('%-d')}–{end_date.strftime('%-d %B %Y')}*"
    )
    print("✅ Posted summary CSV to Slack.")

    return df_summary


def main():
    load_dotenv()

    today = datetime.today()
    this_monday = today - timedelta(days=today.weekday())  # Gets the Monday of current week

    # If today is the 1st of a month, run summary for the full previous month
    if today.day == 1:
        last_day_prev_month = today - timedelta(days=1)
        reference_monday = last_day_prev_month.replace(day=1)
        print(f"Running full-month clustering for {reference_monday.strftime('%Y-%m-%d')} to {last_day_prev_month.strftime('%Y-%m-%d')}")
        run_weekly_clustering(reference_monday, custom_end_date=last_day_prev_month)
    else:
        print(f"Running weekly clustering for {this_monday.strftime('%Y-%m-%d')}")
        run_weekly_clustering(this_monday)

    print("🟨 Today:", today)
    print("🟨 Is 1st of month?", today.day == 1)
    if today.day == 1:
        print("🟨 Running monthly clustering...")
        print("🟨 Reference start date:", reference_monday)
        print("🟨 End date:", last_day_prev_month)


if __name__ == "__main__":
    main()
