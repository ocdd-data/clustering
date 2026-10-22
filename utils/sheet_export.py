import csv

import pandas as pd

# Clusters as they appear in the transition matrices: 0-4, then X (no rides that month).
CLUSTERS = ["0", "1", "2", "3", "4", "X"]


def write_sheet_import(path, region, month_start, df_curr, count_path, percent_path):
    """One CSV with everything the user base segmentation sheet takes for a month, for its Apps Script.

    Columns A:D are section / from / to / value:
      riders    cluster 0-4 and "total"   riders in each cluster this month
      counts    from 0-4,X  to 0-4,X       transition count matrix (rows: last month's cluster)
      percents  from 0-4,X  to 0-4,X       transition percentage matrix (each row sums to 100)
    Rows 1-2 carry the tags the sheet checks (F1:G2): report = "<region> Segmentation" and month_start.
    """
    counts = pd.read_csv(count_path, index_col=0)
    percents = pd.read_csv(percent_path, index_col=0)
    for m in (counts, percents):
        m.index = m.index.astype(str)
        m.columns = m.columns.astype(str)

    totals = df_curr["cluster"].astype(str).value_counts()
    rows = [["section", "from", "to", "value"]]
    rows += [["riders", k, "", int(totals.get(k, 0))] for k in CLUSTERS[:5]]
    rows.append(["riders", "total", "", int(len(df_curr))])
    for section, m in (("counts", counts), ("percents", percents)):
        for a in CLUSTERS:
            for b in CLUSTERS:
                value = m.loc[a, b] if a in m.index and b in m.columns else 0
                rows.append([section, a, b, value])

    rows[0] += ["", "report", f"{region} Segmentation"]
    rows[1] += ["", "month_start", month_start]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    return path
