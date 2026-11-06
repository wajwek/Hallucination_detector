"""Describe SGI distributions without changing scores or decision rules."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

TASKS = ("qa", "dialogue", "summarization", "all")
QUANTILES = {"p01": .01, "p05": .05, "p25": .25, "median": .50,
             "p75": .75, "p95": .95, "p99": .99, "p999": .999}


def distribution_summary(frame):
    """Return class-wise sample statistics for finite SGI and binary labels."""
    required = {"subset", "label", "sgi"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Missing score columns: {sorted(missing)}")
    labels = pd.to_numeric(frame.label, errors="coerce")
    scores = pd.to_numeric(frame.sgi, errors="coerce")
    clean = frame.loc[np.isfinite(scores) & labels.isin([0, 1]), ["subset"]].copy()
    clean["label"] = labels.loc[clean.index].astype(int)
    clean["sgi"] = scores.loc[clean.index]
    rows = []
    for task in TASKS:
        group = clean if task == "all" else clean.loc[clean.subset == task]
        for label in [0, 1]:
            values = group.loc[group.label == label, "sgi"]
            quantiles = {name: values.quantile(q, interpolation="linear")
                         for name, q in QUANTILES.items()}
            rows.append(dict(subset=task, label=label,
                             response_class="grounded" if label == 0 else "hallucinated",
                             count=len(values), mean=values.mean(), std=values.std(ddof=1),
                             minimum=values.min(), **quantiles, maximum=values.max(),
                             iqr=quantiles["p75"]-quantiles["p25"]))
    return pd.DataFrame(rows), clean


def plot_distributions(clean, output_dir):
    """Save central histograms and full ECDFs, with visible window exclusions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    windows = []
    for task in TASKS:
        group = clean if task == "all" else clean.loc[clean.subset == task]
        if group.empty:
            continue
        lower, upper = group.sgi.quantile([.005, .995]).to_numpy()
        if upper <= lower:
            margin = max(abs(lower)*.05, .1)
            lower, upper = lower-margin, upper+margin
        edges = np.linspace(lower, upper, 61)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
        for label, name, color in [(0, "Grounded", "#2166ac"), (1, "Hallucinated", "#b2182b")]:
            values = group.loc[group.label == label, "sgi"].to_numpy()
            if not len(values):
                continue
            outside = int(np.sum((values < lower) | (values > upper)))
            axes[0].hist(values, bins=edges, weights=np.full(len(values),100/len(values)),
                         histtype="step", linewidth=1.8, color=color,
                         label=f"{name} (outside window: {outside}/{len(values)})")
            ordered = np.sort(values)
            axes[1].step(ordered, np.arange(1,len(values)+1)/len(values), where="post",
                         color=color, linewidth=1.6, label=name)
            windows.append(dict(subset=task, label=label, plot_lower=lower, plot_upper=upper,
                                total_rows=len(values), outside_window_rows=outside))
        axes[0].set(title="Central window: pooled 0.5th-99.5th percentiles",
                    xlabel="SGI", ylabel="% of all class responses per bin", xlim=(lower,upper))
        axes[1].set_xscale("log" if (group.sgi > 0).all() else "symlog")
        axes[1].set(title="Full distribution: cumulative fraction",
                    xlabel="SGI (log scale)", ylabel="Fraction with SGI at or below x", ylim=(0,1.01))
        if not (group.sgi > 0).all():
            axes[1].set_xlabel("SGI (symmetric log scale)")
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8,loc="best")
        fig.suptitle(f"HaluEval: {task}",fontsize=12)
        # No timestamps or host-specific metadata are embedded in the figure.
        fig.savefig(output_dir/f"sgi_distribution_{task}.png",dpi=170,
                    metadata={"Software":"SGI evaluation"})
        plt.close(fig)
    return pd.DataFrame(windows)


def distribution_report(stats):
    """Render measured statistics and limited, task-specific observations."""
    lines = ["## Distribution analysis", "",
             "This step describes SGI before investigating individual examples or denominator angles. "
             "All statistics use finite scores and binary labels. Standard deviations use the sample definition (ddof=1); "
             "quantiles use linear interpolation. No scores are removed to make the distributions look cleaner.", "",
             "| Task | Class | N | Mean | Median | Sample SD | p95 | p99 | Maximum |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in stats.itertuples():
        lines.append(f"| {row.subset} | {row.response_class} | {row.count:,} | {row.mean:.6f} | "
                     f"{row.median:.6f} | {row.std:.6f} | {row.p95:.6f} | {row.p99:.6f} | {row.maximum:.6f} |")
    def values(task,label):
        return stats.loc[(stats.subset==task)&(stats.label==label)].iloc[0]
    qg,qh=values("qa",0),values("qa",1)
    dg,dh=values("dialogue",0),values("dialogue",1)
    sg,sh=values("summarization",0),values("summarization",1)
    lines += ["", "### What the distributions show", "",
              f"**QA:** the reference median is {qg['median']:.6f}, compared with {qh['median']:.6f} for hallucinations. "
              f"The middle 50% ranges are [{qg.p25:.6f}, {qg.p75:.6f}] and [{qh.p25:.6f}, {qh.p75:.6f}]. "
              "These intervals describe spread, not confidence limits. Their overlap shows why a single cutoff still makes errors.", "",
              f"**Dialogue:** the reference mean is {dg['mean']:.6f}, but its median is {dg['median']:.6f}. "
              f"The reference maximum is {dg.maximum:.6f} and sample SD is {dg['std']:.6f}, "
              f"compared with {dh.maximum:.6f} and {dh['std']:.6f} for hallucinations. "
              f"The hallucination median is {dh['median']:.6f}. "
              "The mean and median orders disagree in the recorded data. The reference distribution has a long upper tail; "
              "its mean is not a good description of a typical response. This helps interpret the small standardized mean difference, "
              "but does not identify why those extreme ratios occur. Inspecting their denominator angles remains a separate next step.", "",
              f"**Summarization:** reference and hallucination medians are {sg['median']:.6f} and {sh['median']:.6f}. "
              f"Their middle 50% ranges are [{sg.p25:.6f}, {sg.p75:.6f}] and [{sh.p25:.6f}, {sh.p75:.6f}]. "
              "In the recorded run, both the means and medians put hallucinated summaries higher. "
              "The reversed direction therefore also appears in the central values; it is not merely a difference between the two maxima. "
              "This step does not isolate the effects of the fixed instruction, context truncation or response style.", "",
              "**Pooled scores:** task-specific locations differ substantially. The pooled distribution mixes these locations and should not replace the separate task comparisons.", "",
              "### Distribution figures", "",
              "The left panel shows a shared within-task window from the pooled 0.5th to 99.5th percentile. "
              "Each histogram bin is a percentage of all usable responses in that class; the legend counts values outside the displayed window. "
              "The right panel shows the complete empirical cumulative distribution, including extremes, on a log score axis. "
              "A cumulative value of 0.8 at x means 80% of that class has SGI at most x. "
              "Neither display changes the data or metrics.", ""]
    for task in TASKS:
        lines += [f"![SGI distributions for {task}](results/HaluEval/distributions/sgi_distribution_{task}.png)",""]
    lines += ["[Full class statistics](results/HaluEval/distributions/sgi_distributions.csv) include minimum, p01, p05, p25, median, p75, p95, p99, p99.9, maximum and interquartile range. "
              "[Plot window counts](results/HaluEval/distributions/plot_windows.csv) make visual clipping explicit. "
              "[Distribution settings](results/HaluEval/distributions/distribution_config.json) record the source-score hash and plotting choices.",""]
    return "\n".join(lines)


def analyze_distributions(frame, output_dir, scores_sha256):
    stats, clean = distribution_summary(frame)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    stats.to_csv(output_dir/"sgi_distributions.csv",index=False)
    windows=plot_distributions(clean,output_dir)
    windows.to_csv(output_dir/"plot_windows.csv",index=False)
    settings=dict(scores_sha256=scores_sha256, rows=len(frame), evaluated_rows=len(clean),
                  excluded_rows=len(frame)-len(clean), sample_sd_ddof=1, quantiles=QUANTILES,
                  quantile_interpolation="linear", plot_window_quantiles=[.005,.995],
                  histogram_bins=60, histogram_weights="percent of all valid class responses",
                  full_range_plot="ECDF; log SGI axis for positive scores", score_transform="none")
    (output_dir/"distribution_config.json").write_text(json.dumps(settings,indent=2)+"\n")
    return stats,distribution_report(stats)
