import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def generate_cluster_transition_barchart(df_prev, df_curr, prev_month_label, curr_month_label, output_dir):
    df_prev = df_prev.rename(columns={"cluster": "cluster_prev"})
    df_curr = df_curr.rename(columns={"cluster": "cluster_curr"})

    merged = pd.merge(
        df_prev[["rider_uuid", "cluster_prev"]],
        df_curr[["rider_uuid", "cluster_curr"]],
        on="rider_uuid", how="outer"
    )

    merged["cluster_prev"] = merged["cluster_prev"].fillna("X").astype(str)
    merged["cluster_curr"] = merged["cluster_curr"].fillna("X").astype(str)

    count_matrix = merged.groupby(["cluster_prev", "cluster_curr"]).size().unstack(fill_value=0)
    percent_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0) * 100

    os.makedirs(output_dir, exist_ok=True)

    count_path = os.path.join(output_dir, f"transition_counts_{curr_month_label}.csv")
    percent_path = os.path.join(output_dir, f"transition_percents_{curr_month_label}.csv")
    chart_path = os.path.join(output_dir, f"cluster_transition_chart_{curr_month_label}.png")

    count_matrix.to_csv(count_path)
    percent_matrix.to_csv(percent_path)

    clusters_prev = sorted(count_matrix.index.astype(str))
    pastel_palette = sns.color_palette("pastel", len(clusters_prev))
    color_map = dict(zip(clusters_prev, pastel_palette))

    fig, axes = plt.subplots(2, 1, figsize=(14, 14))

    for ax, data, title, ylabel, is_percent in [
        (axes[0], percent_matrix, f"{prev_month_label} → {curr_month_label} (% Riders)", "% of Riders", True),
        (axes[1], count_matrix, f"{prev_month_label} → {curr_month_label} (Counts)", "Number of Riders", False)
    ]:
        data.plot(kind='bar', stacked=True, ax=ax, color=[color_map[c] for c in clusters_prev], legend=False)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xlabel(f"To Cluster ({curr_month_label})")
        ax.set_xticklabels(data.columns, rotation=0)

        for container in ax.containers:
            labels = []
            for bar in container:
                height = bar.get_height()
                if height < (0.5 if is_percent else 1000): 
                    labels.append("")
                else:
                    label = f"{height:.2f}%" if is_percent else f"{int(round(height)):,}"
                    labels.append(label)

            ax.bar_label(
                container,
                labels=labels,
                label_type='center',
                fontsize=8,
                color='black'
            )

        handles = [plt.Rectangle((0, 0), 1, 1, color=color_map[c]) for c in clusters_prev]
        ax.legend(handles, clusters_prev, title=f"From ({prev_month_label})", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)

    return chart_path, count_path, percent_path
