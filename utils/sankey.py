import os
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

def generate_cluster_transition_sankey(df_prev, df_curr, output_dir, month_label, slack):
    cluster_names = {
        0: 'Cluster 0 - New or Inactive Riders',
        1: 'Cluster 1 - Occasional Riders',
        2: 'Cluster 2 - Moderately Active Passengers',
        3: 'Cluster 3 - Regular Riders',
        4: 'Cluster 4 - Highly Active Passengers'
    }

    df_prev = df_prev.rename(columns={'cluster': 'cluster_prev'})
    df_curr = df_curr.rename(columns={'cluster': 'cluster_curr'})

    merged = pd.merge(
        df_prev[['rider_uuid', 'cluster_prev']],
        df_curr[['rider_uuid', 'cluster_curr']],
        on='rider_uuid', how='inner'
    )

    total_prev = df_prev['cluster_prev'].value_counts()
    total_curr = df_curr['cluster_curr'].value_counts()
    n_prev = len(df_prev)
    n_curr = len(df_curr)

    cluster_ids = sorted(set(total_prev.index).union(total_curr.index))
    label_base = {cid: cluster_names[cid] for cid in cluster_ids}

    # Format: Cluster (Month) XX%
    label_list = [f"{label_base[cid]} (Prev) {total_prev.get(cid, 0)/n_prev:.0%}" for cid in cluster_ids] + \
                 [f"{label_base[cid]} (Curr) {total_curr.get(cid, 0)/n_curr:.0%}" for cid in cluster_ids]
    label_idx = {label: i for i, label in enumerate(label_list)}

    flow = merged.groupby(['cluster_prev', 'cluster_curr']).size().reset_index(name='count')
    flow['source_label'] = flow['cluster_prev'].map(lambda cid: f"{label_base[cid]} (Prev) {total_prev.get(cid, 0)/n_prev:.0%}")
    flow['target_label'] = flow['cluster_curr'].map(lambda cid: f"{label_base[cid]} (Curr) {total_curr.get(cid, 0)/n_curr:.0%}")
    flow['source'] = flow['source_label'].map(label_idx)
    flow['target'] = flow['target_label'].map(label_idx)
    flow['label'] = flow.apply(
        lambda r: f"{r['count']:,} riders from {label_base[r['cluster_prev']]} to {label_base[r['cluster_curr']]}", axis=1
    )

    palette = pc.qualitative.Set3
    cluster_color_map = {cid: palette[cid % len(palette)] for cid in cluster_ids}
    node_colors = [cluster_color_map[cid] for cid in cluster_ids] * 2
    link_colors = [cluster_color_map[cid] for cid in flow['cluster_prev']]

    fig = go.Figure(go.Sankey(
        arrangement='snap',
        node=dict(
            label=label_list,
            color=node_colors,
            pad=20,
            thickness=30,
            line=dict(color='black', width=0.6)
        ),
        link=dict(
            source=flow['source'],
            target=flow['target'],
            value=flow['count'],
            color=link_colors,
            customdata=flow['label'],
            hovertemplate="%{customdata}<extra></extra>"
        )
    ))

    fig.update_layout(
        title_text=f"Rider Cluster Transition ({month_label})",
        font_size=14,
        height=1000,
        width=1400,
        margin=dict(l=40, r=40, t=100, b=40)
    )

    os.makedirs(output_dir, exist_ok=True)
    slug = month_label.replace(" ", "_").replace("→", "to")
    path_png = os.path.join(output_dir, f"cluster_transition_sankey_{slug}.png")
    path_html = path_png.replace(".png", ".html")

    fig.write_html(path_html)
    fig.write_image(path_png, width=1400, height=1000)

    slack.uploadFile(path_png, os.getenv("SLACK_CHANNEL"), f"*Cluster Transition Sankey* ({month_label})")
    slack.uploadFile(path_html, os.getenv("SLACK_CHANNEL"), f"<{path_html}|Interactive Chart> ({month_label})")
