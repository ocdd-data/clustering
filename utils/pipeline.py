from pathlib import Path
import pandas as pd
from utils.helpers import Query, Redash
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import numpy as np
import os
import calendar

def add_month_length_normalized_features(df: pd.DataFrame, month_fallback: str = None) -> pd.DataFrame:
    df = df.copy()

    if 'month' in df.columns:
        df['days_in_month'] = pd.to_datetime(df['month']).dt.days_in_month

    elif month_fallback:
        df['days_in_month'] = pd.to_datetime(month_fallback).days_in_month

    else:
        # fallback: assume December 2024 (based on query)
        year, month = 2024, 12
        df['days_in_month'] = calendar.monthrange(year, month)[1]

    df['count_rate'] = df['count'] / df['days_in_month']
    df['trip_rate'] = df['trip'] / df['days_in_month']
    return df


ORDERED_CLUSTER_MAPPING = {
    0: 'Cluster 0 - New or Inactive Riders',
    1: 'Cluster 1 - Occasional Riders',
    2: 'Cluster 2 - Moderately Active Riders',
    3: 'Cluster 3 - Regular Riders',
    4: 'Cluster 4 - Highly Active Riders'
}

class RiderClusterTrainer:
    def __init__(self, n_clusters=5):
        self.k = n_clusters
        self.features = ['count_rate', 'trip_rate', 'avg']
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init='auto')

    ORDERED_CLUSTER_MAPPING = {
        0: 'Cluster 0 - New or Inactive Riders',
        1: 'Cluster 1 - Occasional Riders',
        2: 'Cluster 2 - Moderately Active Riders',
        3: 'Cluster 3 - Regular Riders',
        4: 'Cluster 4 - Highly Active Riders'
    }
    def train(self, df: pd.DataFrame, month: str = None):
        df = add_month_length_normalized_features(df, month_fallback=month)
        df = df.dropna(subset=self.features)
        X_scaled = self.scaler.fit_transform(df[self.features])
        self.kmeans.fit(X_scaled)

        centroids = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.features)
        centroids['original_cluster'] = centroids.index
        centroids['activity_score'] = centroids[self.features].sum(axis=1)

        centroids = centroids.sort_values(by='trip_rate').reset_index(drop=True)

        centroids['Cluster'] = range(len(centroids))
        centroids['Description'] = centroids['Cluster'].map(ORDERED_CLUSTER_MAPPING)

        raw_to_new = dict(zip(centroids['original_cluster'], centroids['Cluster']))

        centroids['Centroid'] = centroids[self.features].values.tolist()
        cluster_def_path = Path("models") / os.getenv("REGION")
        cluster_def_path.mkdir(parents=True, exist_ok=True)

        centroids[['Cluster', 'Description', 'Centroid']].to_csv(cluster_def_path / "cluster_centroids.csv", index=False)

        scaler_df = pd.DataFrame({
            'Feature': self.features,
            'Mean': self.scaler.mean_,
            'Variance': self.scaler.var_
        })
        scaler_df.to_csv(cluster_def_path / "scaler_params.csv", index=False)

        joblib.dump(self.kmeans, cluster_def_path / "kmeans_model.pkl")
        joblib.dump(self.scaler, cluster_def_path / "scaler.pkl")
        
def main():
    load_dotenv()

    region = os.getenv("REGION")
    query_id = int(os.getenv("QUERY_ID"))

    redash = Redash(
        key=os.getenv("REDASH_API_KEY"),
        base_url=os.getenv("REDASH_BASE_URL")
    )

    query = Query(query_id)  
    redash.run_queries([query])
    df = redash.get_result(query.id)

    trainer = RiderClusterTrainer()
    trainer.train(df, month=os.getenv("START_DATE"))

    print(f"Model saved to models/{region}")

if __name__ == "__main__":
    main()
