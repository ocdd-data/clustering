from pathlib import Path
import pandas as pd
from utils.helpers import Query, Redash
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib
import numpy as np
import os

ORDERED_CLUSTER_MAPPING = {
    0: 'Cluster 0 - New or Inactive Riders',
    1: 'Cluster 1 - Occasional Riders',
    2: 'Cluster 2 - Moderately Active Passengers',
    3: 'Cluster 3 - Regular Riders',
    4: 'Cluster 4 - Highly Active Passengers'
}

class RiderClusterTrainer:
    def __init__(self, n_clusters=5):
        self.k = n_clusters
        self.features = ['count', 'trip', 'avg']
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.k, random_state=42, n_init='auto')

    def train(self, df: pd.DataFrame):
        df = df.dropna(subset=self.features)
        X_scaled = self.scaler.fit_transform(df[self.features])
        self.kmeans.fit(X_scaled)

        centroids = pd.DataFrame(self.kmeans.cluster_centers_, columns=self.features)
        centroids['Cluster'] = centroids.index
        centroids['activity_score'] = centroids[self.features].sum(axis=1)

        sorted_clusters = centroids.sort_values(by='activity_score').reset_index(drop=True)
        description_map = {
            row['Cluster']: ORDERED_CLUSTER_MAPPING[i]
            for i, row in sorted_clusters.iterrows()
        }
        centroids['Description'] = centroids['Cluster'].map(description_map)

        region_dir = Path("models") / os.getenv("REGION")
        region_dir.mkdir(parents=True, exist_ok=True)

        centroids['Centroid'] = centroids[self.features].values.tolist()
        centroids[self.features + ['Cluster', 'Description']].to_csv(region_dir / "cluster_centroids.csv", index=False)

        scaler_df = pd.DataFrame({
            'Feature': self.features,
            'Mean': self.scaler.mean_,
            'Variance': self.scaler.var_
        })
        scaler_df.to_csv(region_dir / "scaler_params.csv", index=False)

        joblib.dump(self.kmeans, region_dir / "kmeans_model.pkl")
        joblib.dump(self.scaler, region_dir / "scaler.pkl")


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
    trainer.train(df)

    print(f"Model saved to models/{region}")

if __name__ == "__main__":
    main()
