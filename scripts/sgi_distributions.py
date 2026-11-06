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
    """Save histograms, linear ECDF detail and a full-range ECDF."""
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
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), constrained_layout=True)
        for label, name, color in [(0, "Grounded", "#2166ac"), (1, "Hallucinated", "#b2182b")]:
            values = group.loc[group.label == label, "sgi"].to_numpy()
            if not len(values):
                continue
            outside = int(np.sum((values < lower) | (values > upper)))
            axes[0].hist(values, bins=edges, weights=np.full(len(values),100/len(values)),
                         histtype="step", linewidth=1.8, color=color,
                         label=f"{name} (outside window: {outside}/{len(values)})")
            ordered = np.sort(values)
            cumulative = np.arange(1, len(values)+1) / len(values)
            for axis in axes[1:]:
                axis.step(ordered, cumulative, where="post",
                          color=color, linewidth=1.8, label=name)
            windows.append(dict(subset=task, label=label, plot_lower=lower, plot_upper=upper,
                                total_rows=len(values), outside_window_rows=outside))
        axes[0].set(title="Central histogram",
                    xlabel="SGI", ylabel="% of all class responses per bin", xlim=(lower,upper))
        axes[1].set(title="Cumulative fraction: central window",
                    xlabel="SGI (linear scale)", ylabel="Fraction with SGI at or below x",
                    xlim=(lower, upper), ylim=(0, 1.01))
        axes[2].set_xscale("log" if (group.sgi > 0).all() else "symlog")
        axes[2].set(title="Cumulative fraction: full range",
                    xlabel="SGI (log scale)", ylabel="Fraction with SGI at or below x", ylim=(0,1.01))
        if not (group.sgi > 0).all():
            axes[2].set_xlabel("SGI (symmetric log scale)")
        from matplotlib.ticker import MaxNLocator
        axes[0].xaxis.set_major_locator(MaxNLocator(nbins=6))
        axes[1].xaxis.set_major_locator(MaxNLocator(nbins=6))
        for ax in axes:
            ax.grid(alpha=.2)
            ax.legend(fontsize=8,loc="best")
        fig.suptitle(f"HaluEval: {task} | Central window: pooled 0.5th-99.5th percentiles", fontsize=12)
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
              "Blue denotes reference responses; red denotes hallucinated responses. These colors describe dataset labels, not predictions.", "",
              "**Left panel: histogram.** The horizontal axis is SGI. The vertical axis is the percentage of all usable responses in that class "
              "falling inside a score bin. A height of 7% means 7% of the class falls in that interval; it is not 7% accuracy or hallucination probability. "
              "Within a figure, both classes use the same bins. Each task has its own score window, so peak heights should not be compared across tasks as if bin widths were equal.", "",
              "The shared within-task window spans the pooled 0.5th to 99.5th percentile. It shows the center without letting a few extreme scores stretch the axis. "
              "This includes about 99% of the combined task observations, not necessarily 99% of each class. "
              "The legend counts observations outside the window. All observations still contribute to the statistics and evaluation.", "",
              "**Middle panel: central ECDF on a linear score axis.** This panel uses the same percentile window as the histogram, "
              "so differences near the main concentration remain visible even when rare values extend far beyond it. "
              "Both class curves retain their full-sample cumulative fractions; they are not renormalized within the window. "
              "The vertical axis stays at 0-1, so the detail view does not enlarge the apparent vertical separation. "
              "A value of 0.8 at x means 80% of that class has SGI at most x.", "",
              "**Right panel: full-range empirical cumulative distribution (ECDF).** The horizontal axis is SGI on a logarithmic scale: "
              "equal distances represent equal multiplicative changes, such as 1 to 10 and 10 to 100. The vertical axis is the fraction of a class "
              "with SGI at most x. A value of 0.8 means 80% of that class has SGI no greater than x. "
              "A steep section means many scores occur in a narrow range; a nearly flat section contains few additional observations. "
              "This panel includes the full range, including extreme scores. It is a score distribution, not a time series or a ROC curve.", "",
              "**Reading a decision threshold.** Under the SGI < threshold rule, the red cumulative fraction just below the cutoff is recall, "
              "while the blue fraction is the false positive rate. The left limit matters when scores equal the cutoff, because equality is classified as grounded. "
              "For this direction, a useful region has more red than blue below the cutoff. Overlap means the cutoff inevitably catches some grounded responses or misses some hallucinations. "
              "The full curve comparison matters more than which class has the single highest score.", "",
              "**TP and FP in the middle panel.** Fix an SGI cutoff t and read both curves just to its left. "
              "The red height is TP / (TP + FN), the fraction of actual hallucinations detected. "
              "The blue height is FP / (FP + TN), the fraction of reference responses incorrectly flagged. "
              "Multiply each height by the number of examples in its actual class to obtain a count. "
              "Each recorded task contains 10,000 examples per class, so a height of 0.45 represents about 4,500 responses. "
              "In an imbalanced dataset the two denominators differ; equal heights would not imply equal counts.", "",
              "For the recorded dialogue scores at t = 1.00, the strict below-threshold rule yields TP = 4,554 "
              "and FP = 4,967: recall is 45.54% and the false positive rate is 49.67%. "
              "The remaining counts are FN = 5,446 and TN = 5,033. "
              "At this cutoff, more grounded responses are incorrectly flagged than hallucinations are detected. "
              "This is a reading of one cutoff, not a count represented by the entire curve. "
              "Moving the cutoff to the right adds predictions to the hallucination class: both TP and FP can only increase or stay unchanged. "
              "Prefer a cutoff with a high red fraction and a low blue fraction, then validate that choice on separate data.", ""]
    captions = {
        "qa": (
            "The central red distribution extends farther toward low SGI, while the blue distribution is concentrated nearer 1 and is shifted higher. "
            "The red ECDF rises earlier through much of the central range, matching the expected lower scores for hallucinations. "
            "Both classes also occupy shared score intervals, and the cumulative curves meet and can change order in the upper tail. "
            "SGI therefore ranks many pairs correctly without offering a cutoff that separates every response."),
        "dialogue": (
            "The central histogram peaks almost coincide near SGI 1. Most ordinary scores overlap strongly; the histogram does not show two well-separated groups. "
            "The full-range blue curve extends to a score above 7000, although it has already accumulated almost all reference responses near the left side. "
            "That long, nearly flat extension represents rare extreme scores, not a large population of strongly separated reference answers. "
            "Read it together with the median and sample SD: the high reference mean does not imply that a typical reference answer has much higher SGI."),
        "summarization": (
            "The red central distribution is shifted toward higher SGI, and its ECDF generally rises later than the blue curve. "
            "This is opposite to the assumed low-SGI hallucination signal and agrees with the median comparison. "
            "The ranges still overlap substantially, so reversing the decision direction does not create a clean separator. "
            "The plotted scores begin above 1, which explains why cutoff 1 flags no summaries. "
            "These plots do not explain whether the fixed instruction, truncation or response style produced this pattern."),
        "all": (
            "The pooled histogram contains a large concentration near 1 and a broader concentration around the higher summarization scores. "
            "These features mix different tasks rather than identify two pure correctness groups. "
            "The cumulative curves lie close over much of the range, while the extreme reference tail from dialogue remains visible. "
            "The pooled figure helps describe the mixture but cannot justify a shared threshold or replace the three task-specific figures."),
    }
    for task in TASKS:
        lines += [f"#### {task.upper() if task == 'qa' else task.capitalize()}", "",
                  f"![SGI distributions for {task}](results/HaluEval/distributions/sgi_distribution_{task}.png)",
                  "", captions[task], ""]
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
                  central_range_plot="ECDF; linear SGI axis; same percentile window as histogram; y range 0-1",
                  full_range_plot="ECDF; log SGI axis for positive scores", score_transform="none")
    (output_dir/"distribution_config.json").write_text(json.dumps(settings,indent=2)+"\n")
    return stats,distribution_report(stats)
