import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

def generate_cluster_transition_barchart(df_prev, df_curr, prev_month_label, curr_month_label, output_dir, region):
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
    percent_matrix_row = count_matrix.div(count_matrix.sum(axis=1), axis=0) * 100

    os.makedirs(output_dir, exist_ok=True)

    count_path = os.path.join(output_dir, f"transition_counts_{region}_{curr_month_label}.csv")
    percent_path = os.path.join(output_dir, f"transition_percents_{region}_{curr_month_label}.csv")
    chart_path = os.path.join(output_dir, f"cluster_transition_chart_{region}_{curr_month_label}.png")

    count_matrix.to_csv(count_path)
    percent_matrix_row.to_csv(percent_path)

    custom_colors = {
        "0": "#E76F51",
        "1": "#D4A373",
        "2": "#E9C46A",
        "3": "#81B29A",
        "4": "#457B9D",
        "X": "#D4A373"
    }

    unique_clusters = sorted(set(count_matrix.index.astype(str)) | set(count_matrix.columns.astype(str)))
    color_map = {c: custom_colors.get(c, "#CCCCCC") for c in unique_clusters}

    fig, axes = plt.subplots(1, 2, figsize=(36, 12))

    for ax, data, title_suffix, ylabel, is_percent in [
        (axes[0], percent_matrix_row, "(% Riders)", "% of Riders", True),
        (axes[1], count_matrix, "(Counts)", "Number of Riders", False)
    ]:
        bar_width = 0.5
        index = range(len(data.index))
        bar_bottoms = [0] * len(data.index)

        for cluster in data.columns:
            if cluster not in color_map:
                color_map[cluster] = "#CCCCCC"

        for i, row_label in enumerate(data.index):
            row_data = data.loc[row_label]
            for cluster in data.columns:
                value = row_data[cluster]
                if value == 0:
                    continue

                height = value if not (not is_percent and value < 1000) else 1000
                ax.bar(i, height, bottom=bar_bottoms[i],
                       width=bar_width,
                       color=color_map[cluster],
                       edgecolor='white',
                       linewidth=1,
                       label=cluster if i == 0 else "")

                label_val = f"{value:.2f}%" if is_percent else f"{int(round(value)):,}"
                font_size = 6 if (is_percent and value < 0.5) or (not is_percent and value < 1000) else 8
                ax.text(i, bar_bottoms[i] + height / 2, label_val,
                        ha='center', va='center', fontsize=font_size, color='black')
                bar_bottoms[i] += height

        ax.set_xticks(index)
        ax.set_xticklabels(data.index, rotation=0)
        ax.set_xlabel(f"*From Cluster ({prev_month_label})*")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{region} {prev_month_label} → {curr_month_label} {title_suffix}", fontweight='bold')
        ax.grid(axis='y', color='white', linewidth=2)
        ax.set_axisbelow(True)

        if ax == axes[1]:
            legend_labels = sorted(set().union(*[row.sort_values(ascending=False).index for _, row in data.iterrows()]))
            handles = [mpatches.Patch(color=color_map[c], label=c) for c in legend_labels]
            ax.legend(handles=handles, title=f"*To ({curr_month_label})*", bbox_to_anchor=(1.01, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)

    highlight_threshold_count = 30000
    highlight_threshold_percent = 30

    upgrades = []
    downgrades = []
    retained = []
    churned = []
    reactivated = []

    for origin_cluster in count_matrix.index:
        for target_cluster in count_matrix.columns:
            count = count_matrix.loc[origin_cluster, target_cluster]
            percent = percent_matrix_row.loc[origin_cluster].get(target_cluster, 0)

            if count >= highlight_threshold_count or percent >= highlight_threshold_percent:
                try:
                    ori = int(origin_cluster) if origin_cluster != "X" else None
                    tgt = int(target_cluster) if target_cluster != "X" else None
                except:
                    ori, tgt = None, None

                message = (
                    f"*Cluster {origin_cluster} → Cluster {target_cluster}*: {int(count):,} riders ({percent:.1f}% of Cluster {origin_cluster})"
                )

                if ori is not None and tgt is not None:
                    if tgt > ori:
                        upgrades.append(message)
                    elif tgt < ori:
                        downgrades.append(message)
                    else:
                        retained.append(message)
                elif ori is not None and tgt is None:
                    churned.append(message)
                elif ori is None and tgt is not None:
                    reactivated.append(message)

    sections = []
    if upgrades:
        sections.insert(0, "\n".join(f"> {line}" for line in [":rocket: *Upgrades*"] + upgrades))
    if downgrades:
        sections.append("\n".join(f"> {line}" for line in [":boom: *Downgrade*"] + downgrades))
    if retained:
        sections.append("\n".join(f"> {line}" for line in [":repeat: *Retained*"] + retained))
    if churned:
        sections.append("\n".join(f"> {line}" for line in [":zzz: *Churned*"] + churned))
    if reactivated:
        sections.append("\n".join(f"> {line}" for line in [":ambulance: *Reactivated*"] + reactivated))

    movement_summary_text = "\n\n".join(sections).strip()

    return chart_path, count_path, percent_path, movement_summary_text, []
