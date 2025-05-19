import os
from datetime import datetime, timedelta

from dotenv import load_dotenv

from utils.helpers import Query, Redash
from utils.slack import SlackBot
from utils.transformer import cluster_riders, summarize_clusters


def main():
    # Load credentials
    load_dotenv()

    # Setup
    region = "SG"
    today = datetime.today()
    first_day_last_month = (today.replace(day=1) - timedelta(days=1)).replace(day=1)
    last_day_last_month = today.replace(day=1) - timedelta(days=1)

    start_date = first_day_last_month.strftime("%Y-%m-%d")
    end_date = last_day_last_month.strftime("%Y-%m-%d")
    output_month = first_day_last_month.strftime("%b_%Y")

    # Instantiate Redash client
    client = Redash(
        key=os.getenv("REDASH_API_KEY"),
        base_url=os.getenv("REDASH_BASE_URL")
    )

    # Pull data
    query = Query(3819, params={
        "Date Range": {"start": start_date, "end": end_date}
    })

    client.run_queries([query])
    df = client.get_result(query.id)

    # Perform Clustering
    k_value = 5
    df_clustered = cluster_riders(df, n_clusters=k_value)
    cluster_summary = summarize_clusters(df_clustered)

    # Save results
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"rider_clusters_{region}_{output_month}.csv")
    df_clustered.to_csv(output_path, index=False)

    # Slack Notification
    slack = SlackBot()
    slack.uploadFile(
        output_path,
        os.getenv("SLACK_CHANNEL"),
        f"Rider Segmentation Completed* for {region} ({output_month})\n```{cluster_summary}```"
    )


if __name__ == "__main__":
    main()
