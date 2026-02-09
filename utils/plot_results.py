import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import os
import argparse
from tqdm import tqdm
from scipy.signal import savgol_filter
import glob
import json


def load_csv_data(data_path, domain=None, filter_exp_classes=None):
    """
    Load data from CSV files in the specified directory.

    Args:
        data_path (str): Path to directory containing CSV files
        domain (str): Domain to filter (e.g., 'minatar', 'gymnax'). If None, loads all.
        filter_exp_classes (list): List of exp_classes to include (None for all).

    Returns:
        dict: Nested dictionary structure: data[env_id][exp_class][seed] = (steps, values)
    """
    data = {}

    # Pattern to match CSV files
    if domain:
        pattern = os.path.join(data_path, f"{domain}_*.csv")
    else:
        pattern = os.path.join(data_path, "*.csv")

    csv_files = glob.glob(pattern)

    if not csv_files:
        print(f"No CSV files found matching pattern: {pattern}")
        return data

    print(f"Found {len(csv_files)} CSV files. Loading data...")

    for csv_file in tqdm(csv_files, desc="Loading CSV files"):
        # Parse filename: {domain}_{exp_class}_{env_id}.csv
        basename = os.path.basename(csv_file)
        parts = basename[:-4].split("_", 2)  # Remove .csv and split into 3 parts

        if len(parts) < 3:
            print(f"Skipping malformed filename: {basename}")
            continue

        file_domain, exp_class, env_id = parts

        # Apply exp_class filter
        if filter_exp_classes is not None and exp_class not in filter_exp_classes:
            continue

        # Read CSV
        df = pd.read_csv(csv_file)

        if "step" not in df.columns:
            print(f"Skipping {basename}: missing 'step' column")
            continue

        # Initialize nested structure
        if env_id not in data:
            data[env_id] = {}
        if exp_class not in data[env_id]:
            data[env_id][exp_class] = {}

        # Extract seed columns (all columns except 'step')
        seed_columns = [col for col in df.columns if col != "step"]

        for seed_col in seed_columns:
            # Extract seed number from column name (e.g., 'seed1' -> '1')
            seed = seed_col.replace("seed", "")

            # Get steps and values, drop NaN
            valid_mask = ~df[seed_col].isna()
            steps = df.loc[valid_mask, "step"].values
            values = df.loc[valid_mask, seed_col].values

            if len(steps) > 0:
                data[env_id][exp_class][seed] = (steps, values)

    return data


def load_normalization_data(normalization_file="atari_human_norm_stats.json"):
    """Load normalization data from the JSON file."""
    if not os.path.exists(normalization_file):
        raise FileNotFoundError(f"Normalization file {normalization_file} not found.")
    with open(normalization_file, "r") as f:
        return json.load(f)


def load_baseline_data(baseline_file="DQN_atari26.json"):
    """Load DQN baseline scores from the JSON file."""
    if not os.path.exists(baseline_file):
        raise FileNotFoundError(f"Baseline file {baseline_file} not found.")
    with open(baseline_file, "r") as f:
        data = json.load(f)

    # Structure: baseline_scores[env_name] = {"random": x, "human": y, "dqn": z}
    baseline_scores = {}
    for score_entry in data["scores"]:
        game_name = score_entry["game"]
        baseline_scores[game_name] = {
            "random": score_entry.get("random"),
            "human": score_entry.get("human"),
            "dqn": score_entry.get("dqn"),
        }

    return baseline_scores


def plot_results(data, args):
    """
    Plot results with smoothing, mean and std fill.

    Args:
        data (dict): The data dictionary loaded from CSVs
        args: Command line arguments
    """
    smoothing_window = args.smoothing_window
    normalize = args.normalize
    max_steps = getattr(args, "max_steps", None)

    # Load normalization data if needed
    normalization_data = None
    baseline_data = None
    if normalize:
        normalization_data = load_normalization_data()
        if getattr(args, "add_baseline", False):
            baseline_data = load_baseline_data()

    env_ids = sorted(data.keys())
    if args.filter_env_ids:
        env_ids = [eid for eid in env_ids if eid in args.filter_env_ids]
    num_envs = len(env_ids)

    if num_envs == 0:
        print("No data to plot.")
        return

    # Determine unique experiment classes and assign colors
    unique_exp_classes = set()
    for env_id in env_ids:
        for exp_class in data[env_id].keys():
            unique_exp_classes.add(exp_class)

    sorted_exp_classes = sorted(list(unique_exp_classes))
    prop_cycle = plt.rcParams["axes.prop_cycle"]
    # colors = prop_cycle.by_key()["color"]
    # exp_class_colors = {
    #     cls: colors[i % len(colors)] for i, cls in enumerate(sorted_exp_classes)
    # }
    algo_colors = get_algorithm_colors()
    color_dict = {alg: algo_colors.get(alg, "#808080") for alg in sorted_exp_classes}
    algo_styles = get_algorithm_linestyles()
    style_dict = {alg: algo_styles.get(alg, "-") for alg in sorted_exp_classes}

    # Create subplots
    cols = 5
    rows = math.ceil(num_envs / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    if num_envs == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for i, env_id in enumerate(env_ids):
        ax = axes[i]
        ax.set_title(env_id, fontsize=args.title_fontsize)
        xlabel = "Frames" if args.use_frames else "Steps"
        ax.set_xlabel(xlabel, fontsize=args.label_fontsize)

        # Set ylabel for the first plot in each row
        if i % cols == 0:
            if args.plot_metric == "iqm":
                ax.set_ylabel(
                    ("Norm. Return" if normalize else "Return"),
                    fontsize=args.label_fontsize,
                )
            else:
                ax.set_ylabel(
                    "Normalized Return" if normalize else "Return",
                    fontsize=args.label_fontsize,
                )

        ax.tick_params(axis="both", which="major", labelsize=args.tick_fontsize)

        env_data = data[env_id]

        # Add DQN baseline line if available
        if baseline_data and env_id in baseline_data:
            baseline_info = baseline_data[env_id]
            dqn_score = baseline_info.get("dqn")
            random_score = baseline_info.get("random")
            human_score = baseline_info.get("human")

            if (
                dqn_score is not None
                and random_score is not None
                and human_score is not None
            ):
                normalized_dqn = (dqn_score - random_score) / (
                    human_score - random_score
                )
                ax.axhline(
                    y=normalized_dqn,
                    color="gray",
                    linestyle=":",
                    linewidth=2,
                    # label="DQN Baseline",
                    alpha=0.7,
                )

        for exp_class in sorted(env_data.keys()):
            exp_runs = env_data[exp_class]

            # Collect data across seeds
            all_interpolated_values = []
            all_step_ranges = []
            all_step_ranges = []

            for seed, (step_vals, metric_vals) in exp_runs.items():
                if len(step_vals) == 0:
                    continue

                # Apply normalization if needed
                if normalize and normalization_data and env_id in normalization_data:
                    rand_score = normalization_data[env_id][0]
                    human_score = normalization_data[env_id][-1]
                    metric_vals = (metric_vals - rand_score) / (
                        human_score - rand_score
                    )

                all_step_ranges.append((step_vals.min(), step_vals.max()))
                all_interpolated_values.append((step_vals, metric_vals))

            # Compute mean and std if there is data
            if all_interpolated_values:
                # Find overall min and max steps
                overall_min = min(s[0] for s in all_step_ranges)
                overall_max = max(s[1] for s in all_step_ranges)
                common_steps = np.linspace(overall_min, overall_max, 500)

                # Interpolate each seed
                interpolated_matrix = []
                for step_vals, metric_vals in all_interpolated_values:
                    interp_val = np.full_like(common_steps, np.nan)
                    valid_mask = (common_steps >= step_vals.min()) & (
                        common_steps <= step_vals.max()
                    )
                    interp_val[valid_mask] = np.interp(
                        common_steps[valid_mask], step_vals, metric_vals
                    )
                    interpolated_matrix.append(interp_val)

                interpolated_matrix = np.array(interpolated_matrix)

                # Compute mean and std (or IQM when requested)
                plot_metric = getattr(args, "plot_metric", "mean")
                if plot_metric == "mean":
                    mean_vals = np.nanmean(interpolated_matrix, axis=0)
                    std_vals = np.nanstd(interpolated_matrix, axis=0)
                else:
                    # Compute per-step Inter-Quartile Mean (IQM): for each time
                    # point, take values within 25-75 percentile and average them.
                    n_steps = interpolated_matrix.shape[1]
                    mean_vals = np.full(n_steps, np.nan)
                    std_vals = np.full(n_steps, np.nan)
                    for t in range(n_steps):
                        vals_t = interpolated_matrix[:, t]
                        valid = vals_t[~np.isnan(vals_t)]
                        if valid.size == 0:
                            continue
                        q1 = np.percentile(valid, 25)
                        q3 = np.percentile(valid, 75)
                        trimmed = valid[(valid >= q1) & (valid <= q3)]
                        if trimmed.size == 0:
                            mean_vals[t] = np.mean(valid)
                            std_vals[t] = np.std(valid)
                        else:
                            mean_vals[t] = np.mean(trimmed)
                            std_vals[t] = np.std(trimmed)

                # Apply Savitzky-Golay smoothing
                valid_idx = ~np.isnan(mean_vals)
                if np.sum(valid_idx) > smoothing_window:
                    mean_vals_smooth = np.full_like(mean_vals, np.nan)
                    mean_vals_smooth[valid_idx] = savgol_filter(
                        mean_vals[valid_idx], smoothing_window, 3
                    )
                    mean_vals = mean_vals_smooth

                    std_vals_smooth = np.full_like(std_vals, np.nan)
                    std_vals_smooth[valid_idx] = savgol_filter(
                        std_vals[valid_idx], smoothing_window, 3
                    )
                    std_vals = std_vals_smooth

                # Apply x-axis limit if specified
                plot_steps = common_steps
                plot_mean_vals = mean_vals
                plot_std_vals = std_vals

                if max_steps is not None:
                    mask = common_steps <= max_steps
                    plot_steps = common_steps[mask]
                    plot_mean_vals = mean_vals[mask]
                    plot_std_vals = std_vals[mask]

                # Plot
                x_multiplier = 4 if args.use_frames else 1
                color = color_dict.get(exp_class)
                linestyle = style_dict.get(exp_class)
                (line,) = ax.plot(
                    plot_steps * x_multiplier,
                    plot_mean_vals,
                    label=exp_class,
                    color=color,
                    linestyle=linestyle,
                )
                ax.fill_between(
                    plot_steps * x_multiplier,
                    plot_mean_vals - plot_std_vals,
                    plot_mean_vals + plot_std_vals,
                    alpha=0.2,
                    color=line.get_color(),
                )

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Add common legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    # Sort legend by label name
    if labels:
        sorted_legend = sorted(zip(handles, labels), key=lambda x: x[1])
        handles, labels = zip(*sorted_legend)

    # Save legend separately as PDF
    export_legend(
        handles,
        labels,
        filename=f"{args.domain}_legend.pdf",
        ncol=len(labels),
        fontsize=args.legend_fontsize,
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    output_file = f"{args.domain}_results_plot.pdf"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_file}")
    # plt.show()


def generate_latex_table(data, args):
    """
    Generate a LaTeX table with environments as rows and exp classes as columns.

    Args:
        data (dict): The data dictionary
        args: Command line arguments
    """
    # Use aggregate_latex_step if provided, otherwise fall back to latex_table_step
    target_step = getattr(args, "aggregate_latex_step", None) or args.latex_table_step
    metric_type = args.latex_metric  # 'mean' or 'iqm'
    normalize = args.normalize

    # Load normalization data if needed
    normalization_data = None
    if normalize:
        normalization_data = load_normalization_data()

    # Collect all exp_classes
    all_exp_classes = set()
    for env_id in data:
        for exp_class in data[env_id]:
            if args.exp_classes is None or exp_class in args.exp_classes:
                all_exp_classes.add(exp_class)

    sorted_exp_classes = sorted(list(all_exp_classes))

    # Collect all env_ids
    env_ids = sorted(data.keys())

    # Structure to store results: results[env_id][exp_class] = {'value': X, 'std': Y}
    results = {}

    for env_id in env_ids:
        results[env_id] = {}

        # Skip if normalization is needed but data is missing
        if normalize and (
            normalization_data is None or env_id not in normalization_data
        ):
            continue

        env_data = data[env_id]

        for exp_class in sorted_exp_classes:
            if exp_class not in env_data:
                results[env_id][exp_class] = None
                continue

            exp_runs = env_data[exp_class]
            scores = []

            for seed, (step_vals, metric_vals) in exp_runs.items():
                if len(step_vals) == 0:
                    continue

                # Find closest step
                idx = (np.abs(step_vals - target_step)).argmin()

                # Average 100 points around target
                start_idx = max(0, idx - 50)
                end_idx = min(len(metric_vals), idx + 50)

                avg_ret = np.mean(metric_vals[start_idx:end_idx])
                if np.isnan(avg_ret):
                    continue

                # Apply normalization if needed
                if normalize and normalization_data and env_id in normalization_data:
                    rand_score = normalization_data[env_id][0]
                    human_score = normalization_data[env_id][-1]
                    avg_ret = (avg_ret - rand_score) / (human_score - rand_score)

                scores.append(avg_ret)

            if scores:
                if metric_type == "mean":
                    central = np.mean(scores)
                elif metric_type == "iqm":
                    # Calculate inter-quartile mean
                    q1 = np.percentile(scores, 25)
                    q3 = np.percentile(scores, 75)
                    iqm_vals = [v for v in scores if q1 <= v <= q3]
                    central = np.mean(iqm_vals) if iqm_vals else np.nan
                else:
                    central = np.mean(scores)

                std = np.std(scores)
                results[env_id][exp_class] = {"value": central, "std": std}
            else:
                results[env_id][exp_class] = None

    # Generate LaTeX table
    latex_lines = []
    metric_label = "IQM" if metric_type == "iqm" else "Mean"
    norm_label = " (Human Normalized)" if normalize else ""
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append(
        f"\\caption{{{metric_label} scores at frame {4*target_step//1000000}M{norm_label}}}"
    )
    latex_lines.append("\\adjustbox{max width=\\textwidth}{")
    latex_lines.append("\\begin{tabular}{l" + "c" * len(sorted_exp_classes) + "}")
    latex_lines.append("\\hline")

    # Header
    header = "Environment & " + " & ".join(sorted_exp_classes) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\hline")

    # Rows
    for env_id in env_ids:
        if env_id not in results or not results[env_id]:
            continue

        # Find best value for this row
        best_value = -np.inf
        for exp_class in sorted_exp_classes:
            if results[env_id].get(exp_class) is not None:
                val = results[env_id][exp_class]["value"]
                if not np.isnan(val) and val > best_value:
                    best_value = val

        # Build row
        row_parts = [env_id]
        for exp_class in sorted_exp_classes:
            if results[env_id].get(exp_class) is not None:
                val = results[env_id][exp_class]["value"]
                std = results[env_id][exp_class]["std"]

                # Format value
                cell_text = f"{val:.2f} $\\pm$ {std:.2f}"

                # Bold if best
                if not np.isnan(val) and abs(val - best_value) < 1e-6:
                    cell_text = f"\\textbf{{{cell_text}}}"

                row_parts.append(cell_text)
            else:
                row_parts.append("-")

        row = " & ".join(row_parts) + " \\\\"
        latex_lines.append(row)

    latex_lines.append("\\hline")

    # Calculate and add aggregate row
    aggregate_scores = {}
    aggregate_stds = {}
    for exp_class in sorted_exp_classes:
        all_scores = []
        for env_id in env_ids:
            if env_id in results and results[env_id].get(exp_class) is not None:
                val = results[env_id][exp_class]["value"]
                if not np.isnan(val):
                    all_scores.append(val)

        if all_scores:
            if metric_type == "iqm":
                q1 = np.percentile(all_scores, 25)
                q3 = np.percentile(all_scores, 75)
                iqm_vals = [v for v in all_scores if q1 <= v <= q3]
                aggregate_scores[exp_class] = np.mean(iqm_vals) if iqm_vals else np.nan
                # Standard deviation of per-environment means
                aggregate_stds[exp_class] = np.std(iqm_vals) if iqm_vals else np.nan
            else:
                aggregate_scores[exp_class] = np.mean(all_scores)
                # Standard deviation of per-environment means
                aggregate_stds[exp_class] = np.std(all_scores)
        else:
            aggregate_scores[exp_class] = np.nan
            aggregate_stds[exp_class] = np.nan

    # Find best aggregate value
    best_aggregate = max(
        [v for v in aggregate_scores.values() if not np.isnan(v)], default=-np.inf
    )

    # Build aggregate row
    agg_row_parts = ["\\textbf{Aggregate}"]
    for exp_class in sorted_exp_classes:
        if exp_class in aggregate_scores and not np.isnan(aggregate_scores[exp_class]):
            val = aggregate_scores[exp_class]
            cell_text = f"{val:.2f} $\\pm$ {aggregate_stds[exp_class]:.2f}"

            # Bold if best
            if abs(val - best_aggregate) < 1e-6:
                cell_text = f"\\textbf{{{cell_text}}}"

            agg_row_parts.append(cell_text)
        else:
            agg_row_parts.append("-")

    agg_row = " & ".join(agg_row_parts) + " \\\\"
    latex_lines.append(agg_row)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("}")  # Close adjustbox

    latex_lines.append("\\label{tab:results}")
    latex_lines.append("\\end{table}")

    # Write to file
    output_file = f"{args.domain}_results_table.tex"
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\nLaTeX table saved to {output_file}")


def get_algorithm_colors():
    """Get fixed color mapping for algorithms."""
    return {
        # Green shades: darker to lighter
        "qrc-spr-orth": "#662E7D",  # dark purple
        "qrc-spr": "#905ca6",  # medium purple
        "qrc": "#9E84A2",  # light purple
        "qrc-spr-K1": "#645af5",  # medium purple
        "qrc-spr-K3": "#23f42a",  # light purple
        "qrc-spr-w10": "#1f16ca",  # light purple
        "qrc-spr-w05": "#28db31",  # lighter purple
        "qrc-spr-aug": "#ed3a3a",  # lighter purple
        # red shades: darker to lighter
        "strq-spr-orth2": "#DC3F60",  # dark red
        "strq-spr-orth": "#E0587A",  # darker red
        "strq-spr": "#E46879",  # medium red
        "strq": "#EC9191",  # light red
        # Red shades: darker to lighter
        "dqn-spr": "#0d47a1",  # dark red
        "dqn-rb1": "#29b8f1",  # light red
    }


def get_algorithm_linestyles():
    """Get fixed line style mapping for algorithms."""
    return {
        # spr-orth -> solid
        "qrc-spr-orth": "-",
        "strq-spr-orth2": "-",
        "strq-spr-orth": "-.",
        # spr -> dashed
        "qrc-spr": "--",
        "strq-spr": "--",
        "dqn-spr": "-",
        # base -> dotted
        "qrc": ":",
        "strq": ":",
        "dqn-rb1": ":",
    }


import seaborn as sns


def _decorate_axis(ax, wrect=10, hrect=10, ticklabelsize="large"):
    """Helper function for decorating plots."""
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=ticklabelsize)
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))
    return ax


def export_legend(handles, labels, filename="legend.pdf", ncol=3, fontsize="xx-large"):
    """Export legend to a separate file."""
    fig_leg = plt.figure(figsize=(len(labels) * 1.5, 0.5))
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis("off")
    ax_leg.legend(
        handles,
        labels,
        ncol=ncol,
        loc="center",
        frameon=False,
        fontsize=fontsize,
    )
    fig_leg.savefig(filename, bbox_inches="tight")
    print(f"Legend saved to {filename}")
    plt.close(fig_leg)


def plot_interval_estimates(
    point_estimates,
    interval_estimates,
    metric_names,
    algorithms=None,
    colors=None,
    color_palette="colorblind",
    max_ticks=4,
    subfigure_width=3.4,
    row_height=0.37,
    xlabel_y_coordinate=-0.1,
    xlabel="Normalized Score",
    ylabel=None,
    title_fontsize="xx-large",
    label_fontsize="x-large",
    tick_fontsize="large",
    legend_fontsize="x-large",
    legend_at_bottom=False,
    save_legend_path=None,
    legend_ncol=None,
    **kwargs,
):
    """Plots various metrics with confidence intervals.

    Args:
      point_estimates: Dictionary mapping algorithm to a list or array of point
        estimates of the metrics to plot.
      interval_estimates: Dictionary mapping algorithms to interval estimates
        corresponding to the `point_estimates`. Typically, consists of stratified
        bootstrap CIs.
      metric_names: Names of the metrics corresponding to `point_estimates`.
      algorithms: List of methods used for plotting. If None, defaults to all the
        keys in `point_estimates`.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object for mapping each method to a
        color.
      max_ticks: Find nice tick locations with no more than `max_ticks`. Passed to
        `plt.MaxNLocator`.
      subfigure_width: Width of each subfigure.
      row_height: Height of each row in a subfigure.
      xlabel_y_coordinate: y-coordinate of the x-axis label.
      xlabel: Label for the x-axis.
      text_size: Text size for labels and ticks.
      legend_at_bottom: Whether to place legend at bottom instead of y-tick labels.
      save_legend_path: Path to save legend separately.
      legend_ncol: Number of columns for the legend.
      **kwargs: Arbitrary keyword arguments.

    Returns:
      fig: A matplotlib Figure.
      axes: `axes.Axes` or array of Axes.
    """

    if algorithms is None:
        algorithms = list(point_estimates.keys())
    num_metrics = len(point_estimates[algorithms[0]])
    figsize = (subfigure_width * num_metrics, row_height * len(algorithms))
    if legend_at_bottom:
        figsize = (figsize[0], figsize[1] + 1.2)  # Add extra height for legend

    fig, axes = plt.subplots(nrows=1, ncols=num_metrics, figsize=figsize)
    if colors is None:
        color_palette = sns.color_palette(color_palette, n_colors=len(algorithms))
        colors = dict(zip(algorithms, color_palette))
    h = kwargs.pop("interval_height", 0.6)

    for idx, metric_name in enumerate(metric_names):
        for alg_idx, algorithm in enumerate(algorithms):
            ax = axes[idx] if num_metrics > 1 else axes
            # Plot interval estimates.
            lower, upper = interval_estimates[algorithm][:, idx]
            ax.barh(
                y=alg_idx,
                width=upper - lower,
                height=h,
                left=lower,
                color=colors[algorithm],
                alpha=0.75,
                label=algorithm,
            )
            # Plot point estimates.
            ax.vlines(
                x=point_estimates[algorithm][idx],
                ymin=alg_idx - (7.5 * h / 16),
                ymax=alg_idx + (6 * h / 16),
                label=None,  # No label to avoid color issues in legend
                color="k",
                alpha=0.5,
            )

        ax.set_yticks(list(range(len(algorithms))))
        ax.xaxis.set_major_locator(plt.MaxNLocator(max_ticks))
        if idx != 0 or legend_at_bottom:
            ax.set_yticks([])
            if idx == 0 and legend_at_bottom and ylabel:
                ax.set_ylabel(ylabel, fontsize=label_fontsize)
        else:
            ax.set_yticklabels(algorithms, fontsize=tick_fontsize)
        ax.set_title(metric_name, fontsize=title_fontsize)
        ax.tick_params(axis="both", which="major")
        _decorate_axis(ax, ticklabelsize=tick_fontsize, wrect=5)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="x", alpha=0.25)

    fig.text(0.4, xlabel_y_coordinate, xlabel, ha="center", fontsize=label_fontsize)

    if legend_at_bottom or save_legend_path:
        ax0 = axes[0] if num_metrics > 1 else axes
        handles, labels = ax0.get_legend_handles_labels()
        # Note: Since we removed label from vlines, handles should only contain barh patches with correct colors.

        # Ensure we have handles for all algorithms in order
        unique_handles = []
        unique_labels = []
        seen = set()

        # Original code order is bottom-up (algorithms plotted 0..N).
        # To match, we filter duplicates.
        # Matplotlib legend order usually matches plotting order.

        for h, l in zip(handles, labels):
            if l not in seen and l in algorithms:
                seen.add(l)
                unique_handles.append(h)
                unique_labels.append(l)

        # Sort legend by label name
        if unique_labels:
            sorted_legend = sorted(
                zip(unique_handles, unique_labels), key=lambda x: x[1]
            )
            unique_handles, unique_labels = zip(*sorted_legend)
            unique_handles = list(unique_handles)
            unique_labels = list(unique_labels)

        # Default ncol
        if legend_ncol is None:
            default_ncol = (
                len(algorithms) if len(algorithms) <= 4 else (len(algorithms) + 1) // 2
            )
            ncol = default_ncol
        else:
            ncol = legend_ncol

        if save_legend_path:
            export_legend(
                unique_handles,
                unique_labels,
                save_legend_path,
                ncol=ncol,
                fontsize=legend_fontsize,
            )

        if legend_at_bottom:
            if not save_legend_path:
                fig.legend(
                    unique_handles,
                    unique_labels,
                    loc="lower center",
                    bbox_to_anchor=(0.5, 0.02),  # Below xlabel
                    ncol=ncol,
                    fontsize=legend_fontsize,
                    frameon=False,
                )
                # Adjust bottom to make room for legend and xlabel
                plt.subplots_adjust(
                    wspace=kwargs.pop("wspace", 0.11), left=0.0, bottom=0.25
                )
            else:
                # Legend saved separately, just adjust for xlabel
                plt.subplots_adjust(
                    wspace=kwargs.pop("wspace", 0.11), left=0.0, bottom=0.15
                )
    else:
        plt.subplots_adjust(wspace=kwargs.pop("wspace", 0.11), left=0.0)

    return fig, axes


def _non_linear_scaling(
    performance_profiles, tau_list, xticklabels=None, num_points=4, log_base=2
):
    """Returns non linearly scaled tau as well as corresponding xticks.

    The non-linear scaling of a certain range of threshold values is proportional
    to fraction of runs that lie within that range.

    Args:
      performance_profiles: A dictionary mapping a method to its performance
        profile, where each profile is computed using thresholds in `tau_list`.
      tau_list: List or 1D numpy array of threshold values on which the profile is
        evaluated.
      xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
      num_points: If `xticklabels` are not passed, then specifices the number of
        indices to be generated on a log scale.
      log_base: Base of the logarithm scale for non-linear scaling.

    Returns:
      nonlinear_tau: Non-linearly scaled threshold values.
      new_xticks: x-axis ticks from `nonlinear_tau` that would be plotted.
      xticklabels: x-axis labels correspond to non-linearly scaled thresholds.
    """

    methods = list(performance_profiles.keys())
    nonlinear_tau = np.zeros_like(performance_profiles[methods[0]])
    for method in methods:
        nonlinear_tau += performance_profiles[method]
    nonlinear_tau /= len(methods)
    nonlinear_tau = 1 - nonlinear_tau

    if xticklabels is None:
        tau_indices = np.int32(
            np.logspace(
                start=0, stop=np.log2(len(tau_list) - 1), base=log_base, num=num_points
            )
        )
        xticklabels = [tau_list[i] for i in tau_indices]
    else:
        tau_as_list = list(tau_list)
        # Find indices of x which are in `tau`
        tau_indices = [tau_as_list.index(x) for x in xticklabels]
    new_xticks = nonlinear_tau[tau_indices]
    return nonlinear_tau, new_xticks, xticklabels


def _thin_ticks(ticks, labels, min_spacing=0.05):
    """Reduce the number of ticks to avoid label overlap on crowded axes."""
    if ticks is None or labels is None:
        return ticks, labels
    if len(ticks) <= 1:
        return ticks, labels

    filtered_ticks = [ticks[0]]
    filtered_labels = [labels[0]]
    last = ticks[0]
    for tick, label in zip(ticks[1:], labels[1:]):
        if tick - last >= min_spacing:
            filtered_ticks.append(tick)
            filtered_labels.append(label)
            last = tick

    if filtered_ticks[-1] != ticks[-1]:
        filtered_ticks.append(ticks[-1])
        filtered_labels.append(labels[-1])

    return filtered_ticks, filtered_labels


def _annotate_and_decorate_axis(
    ax,
    labelsize="x-large",
    ticklabelsize="x-large",
    xticks=None,
    xticklabels=None,
    yticks=None,
    legend=False,
    grid_alpha=0.2,
    legendsize="x-large",
    xlabel="",
    ylabel="",
    wrect=10,
    hrect=10,
    rotation=0,
):
    """Annotates and decorates the plot."""
    ax.set_xlabel(xlabel, fontsize=labelsize)
    ax.set_ylabel(ylabel, fontsize=labelsize)
    if xticks is not None:
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(
            xticklabels, rotation=rotation, ha="right" if rotation > 0 else "center"
        )
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.grid(True, alpha=grid_alpha)
    ax = _decorate_axis(ax, wrect=wrect, hrect=hrect, ticklabelsize=ticklabelsize)
    if legend:
        ax.legend(fontsize=legendsize)
    return ax


def plot_performance_profiles(
    performance_profiles,
    tau_list,
    performance_profile_cis=None,
    use_non_linear_scaling=False,
    ax=None,
    colors=None,
    color_palette="colorblind",
    alpha=0.15,
    figsize=(10, 5),
    xticks=None,
    yticks=None,
    xlabel=r"Normalized Score ($\tau$)",
    ylabel=r"Fraction of runs with score $> \tau$",
    linestyles=None,
    **kwargs,
):
    """Plots performance profiles with stratified confidence intervals.

    Args:
      performance_profiles: A dictionary mapping a method to its performance
        profile, where each profile is computed using thresholds in `tau_list`.
      tau_list: List or 1D numpy array of threshold values on which the profile is
        evaluated.
      performance_profile_cis: The confidence intervals (default 95%) of
        performance profiles evaluated at all threshdolds in `tau_list`.
      use_non_linear_scaling: Whether to scale the x-axis in proportion to the
        number of runs within any specified range.
      ax: `matplotlib.axes` object.
      colors: Maps each method to a color. If None, then this mapping is created
        based on `color_palette`.
      color_palette: `seaborn.color_palette` object. Used when `colors` is None.
      alpha: Changes the transparency of the shaded regions corresponding to the
        confidence intervals.
      figsize: Size of the figure passed to `matplotlib.subplots`. Only used when
        `ax` is None.
      xticks: The list of x-axis tick locations. Passing an empty list removes all
        xticks.
      yticks: The list of y-axis tick locations between 0 and 1. If None, defaults
        to `[0, 0.25, 0.5, 0.75, 1.0]`.
      xlabel: Label for the x-axis.
      ylabel: Label for the y-axis.
      linestyles: Maps each method to a linestyle. If None, then the 'solid'
        linestyle is used for all methods.
      **kwargs: Arbitrary keyword arguments for annotating and decorating the
        figure. For valid arguments, refer to `_annotate_and_decorate_axis`.

    Returns:
      `matplotlib.axes.Axes` object used for plotting.
    """

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    if colors is None:
        keys = performance_profiles.keys()
        color_palette = sns.color_palette(color_palette, n_colors=len(keys))
        colors = dict(zip(list(keys), color_palette))

    if linestyles is None:
        linestyles = {key: "solid" for key in performance_profiles.keys()}

    if use_non_linear_scaling:
        tau_list, xticks, xticklabels = _non_linear_scaling(
            performance_profiles, tau_list, xticks
        )
        xticks, xticklabels = _thin_ticks(xticks, xticklabels)
        if xticklabels is not None:
            xticklabels = [f"{val:g}" for val in xticklabels]
    else:
        xticklabels = xticks

    for method, profile in performance_profiles.items():
        ax.plot(
            tau_list,
            profile,
            color=colors[method],
            linestyle=linestyles[method],
            linewidth=kwargs.pop("linewidth", 2.0),
            label=method,
        )
        if performance_profile_cis is not None:
            if method in performance_profile_cis:
                lower_ci, upper_ci = performance_profile_cis[method]
                ax.fill_between(
                    tau_list, lower_ci, upper_ci, color=colors[method], alpha=alpha
                )

    if yticks is None:
        yticks = [0.0, 0.25, 0.5, 0.75, 1.0]

    # Add rotation for non-linear scaling to prevent label overlap
    rotation = 45 if use_non_linear_scaling else 0

    axis = _annotate_and_decorate_axis(
        ax,
        xticks=xticks,
        yticks=yticks,
        xticklabels=xticklabels,
        xlabel=xlabel,
        ylabel=ylabel,
        rotation=rotation,
        **kwargs,
    )

    if use_non_linear_scaling:
        for tick_label in axis.get_xticklabels():
            tick_label.set_rotation(45)
            tick_label.set_horizontalalignment("right")

    return axis


def plot_aggregate_with_rliable(data, args, target_step=None):
    """Use rliable to compute aggregate metrics and plot interval estimates.

    The function builds a mapping from experiment class (algorithm name) to a
    numpy matrix of shape (num_runs x num_envs). Each column corresponds to an
    environment and contains per-seed averaged returns (NaN-padded). Then it
    computes aggregate estimates (Median, IQM, Mean, Optimality Gap) with
    bootstrap CIs and plots them using rliable's plotting utilities.
    """
    try:
        from rliable import library as rly
        from rliable import metrics
        from rliable import plot_utils
    except Exception:
        print(
            "rliable is not installed. Install via 'pip install rliable' to use this."
        )
        return

    if target_step is None:
        target_step = args.aggregate_metrics or args.latex_table_step
        if target_step is None:
            print(
                "target_step required for rliable plotting (provide --aggregate-metrics)"
            )
            return

    normalize = args.normalize
    normalization_data = None
    if normalize:
        normalization_data = load_normalization_data()

    # Determine exp classes to include
    all_exp_classes = set()
    for env_id in data:
        for exp_class in data[env_id]:
            if args.exp_classes is None or exp_class in args.exp_classes:
                all_exp_classes.add(exp_class)

    sorted_exp_classes = sorted(list(all_exp_classes))
    env_ids = sorted(data.keys())

    if not sorted_exp_classes or not env_ids:
        print("No data available for rliable plotting.")
        return

    # Build score dictionary: alg -> (num_runs x num_envs) matrix
    score_dict = {}
    for exp_class in sorted_exp_classes:
        # collect per-env lists of per-seed averages
        columns = []
        max_runs = 0
        for env_id in env_ids:
            if env_id not in data or exp_class not in data[env_id]:
                columns.append([])
                continue

            per_env_runs = []
            for seed, (step_vals, metric_vals) in data[env_id][exp_class].items():
                if len(step_vals) == 0:
                    continue
                idx = (np.abs(step_vals - target_step)).argmin()
                start_idx = max(0, idx - 50)
                end_idx = min(len(metric_vals), idx + 50)
                avg_ret = np.mean(metric_vals[start_idx:end_idx])

                if normalize and normalization_data and env_id in normalization_data:
                    rand_score = normalization_data[env_id][0]
                    human_score = normalization_data[env_id][-1]
                    avg_ret = (avg_ret - rand_score) / (human_score - rand_score)

                per_env_runs.append(avg_ret)

            columns.append(per_env_runs)
            max_runs = max(max_runs, len(per_env_runs))

        num_envs = len(env_ids)
        if max_runs == 0:
            # No runs for this algorithm: create empty (0 x num_envs) matrix
            score_dict[exp_class] = np.full((0, num_envs), np.nan)
            continue

        mat = np.full((max_runs, num_envs), np.nan)
        for j, col in enumerate(columns):
            for i, val in enumerate(col):
                mat[i, j] = val
            # If this environment has fewer than `max_runs`, fill the remaining
            # rows with the mean of the available runs for that column.
            if len(col) > 0:
                # Compute mean ignoring possible NaNs in `col`.
                col_mean = np.nanmean(col)
                if not np.isnan(col_mean) and len(col) < max_runs:
                    mat[len(col) :, j] = col_mean

        score_dict[exp_class] = mat

    # Aggregate function returning [Median, IQM, Mean, Optimality Gap]
    # Note: We wrap the rliable functions to handle NaN gracefully
    def safe_aggregate_median(x):
        """Compute median using nanmedian for robustness to NaN."""
        # Flatten and remove NaN values
        flat = x.flatten()
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            return np.nan
        return np.median(valid)

    def safe_aggregate_mean(x):
        """Compute mean using nanmean for robustness to NaN."""
        # Flatten and remove NaN values
        flat = x.flatten()
        valid = flat[~np.isnan(flat)]
        if len(valid) == 0:
            return np.nan
        return np.mean(valid)

    def safe_aggregate_optimality_gap(x):
        """Compute optimality gap, handling NaN values."""
        try:
            # The optimality gap typically needs reference scores
            # For now, if there are NaN values, we compute based on valid data only
            # First try the rliable version
            result = metrics.aggregate_optimality_gap(x)
            if np.isnan(result):
                # If it returns NaN, compute manually
                # Optimality gap = 1 - (mean(scores) / max(scores))
                flat = x.flatten()
                valid = flat[~np.isnan(flat)]
                if len(valid) == 0:
                    return np.nan
                mean_score = np.mean(valid)
                max_score = np.max(valid)
                if max_score == 0:
                    return np.nan
                return 1.0 - (mean_score / max_score)
            return result
        except:
            # Fallback computation
            flat = x.flatten()
            valid = flat[~np.isnan(flat)]
            if len(valid) == 0:
                return np.nan
            mean_score = np.mean(valid)
            max_score = np.max(valid)
            if max_score == 0:
                return np.nan
            return 1.0 - (mean_score / max_score)

    if args.domain == "atari" and args.normalize:
        aggregate_func = lambda x: np.array(
            [
                safe_aggregate_median(x),
                metrics.aggregate_iqm(x),  # IQM already handles NaN well
                safe_aggregate_mean(x),
                # safe_aggregate_optimality_gap(x),
            ]
        )
    else:
        aggregate_func = lambda x: np.array(
            [
                safe_aggregate_median(x),
                metrics.aggregate_iqm(x),  # IQM already handles NaN well
                safe_aggregate_mean(x),
            ]
        )

    # Compute per-algorithm IQM and sort algorithms by IQM (descending)
    iqm_scores = {}
    for alg, mat in score_dict.items():
        try:
            # metrics.aggregate_iqm expects a (num_runs x num_envs) array
            val = metrics.aggregate_iqm(mat)
            # Convert NaN to -inf for sorting
            iqm_scores[alg] = np.nan_to_num(val, nan=-np.inf)
        except Exception:
            iqm_scores[alg] = -np.inf

    ordered_algorithms = sorted(
        list(score_dict.keys()), key=lambda k: iqm_scores.get(k, -np.inf), reverse=False
    )

    # Compute interval estimates (use fewer reps by default for speed)
    aggregate_scores, aggregate_score_cis = rly.get_interval_estimates(
        score_dict, aggregate_func, reps=5000
    )

    # Filter out algorithms with all-NaN scores
    valid_algorithms = [
        alg for alg in ordered_algorithms if not np.all(np.isnan(aggregate_scores[alg]))
    ]

    if not valid_algorithms:
        print("No valid algorithms with non-NaN scores. Cannot generate plot.")
        return

    # Filter score dicts to only include valid algorithms
    filtered_scores = {alg: aggregate_scores[alg] for alg in valid_algorithms}
    filtered_cis = {alg: aggregate_score_cis[alg] for alg in valid_algorithms}

    if args.domain == "atari" and args.normalize:
        # metrics_names = ["Median", "IQM", "Mean", "Optimality Gap"]
        metrics_names = ["Median", "IQM", "Mean"]
    else:
        metrics_names = ["Median", "IQM", "Mean"]

    # Get fixed algorithm colors
    algo_colors = get_algorithm_colors()
    color_dict = {alg: algo_colors.get(alg, "#808080") for alg in valid_algorithms}

    fig, axes = plot_interval_estimates(
        filtered_scores,
        filtered_cis,
        metric_names=metrics_names,
        ylabel=args.domain.capitalize(),
        algorithms=valid_algorithms,
        colors=color_dict,
        xlabel_y_coordinate=0.0,
        xlabel=("Human Normalized Score" if normalize else "Score"),
        title_fontsize=args.title_fontsize,
        label_fontsize=args.label_fontsize,
        tick_fontsize=args.tick_fontsize,
        legend_fontsize=args.legend_fontsize,
        legend_at_bottom=args.rliable_legend_bottom,
        save_legend_path=(
            f"{args.domain}_rliable_legend.pdf" if args.save_rliable_legend else None
        ),
        legend_ncol=4,
    )

    # Apply fixed colors to the interval estimates plot
    if isinstance(axes, np.ndarray):
        ax_list = axes.flatten()
    else:
        ax_list = [axes]

    for ax in ax_list:
        for line in ax.get_lines():
            label = line.get_label()
            if label and label in algo_colors:
                line.set_color(algo_colors[label])

    out_file = f"{args.domain}_rliable_aggregate.pdf"
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Rliable aggregate plot saved to {out_file}")

    # Load ALE scores as a dictionary mapping algorithms to their human normalized
    # score matrices, each of which is of size `(num_runs x num_games)`.
    atari_200m_normalized_score_dict = score_dict
    # Human normalized score thresholds
    if args.domain == "atari" and args.normalize:
        atari_200m_thresholds = np.linspace(0.0, 9.0, 81)
    elif args.domain == "atari":
        atari_200m_thresholds = np.linspace(0.0, 6000.0, 81)
    elif args.domain == "minatar":
        atari_200m_thresholds = np.linspace(0.0, 170.0, 81)
    elif args.domain == "octax":
        atari_200m_thresholds = np.linspace(0.0, 280.0, 81)
    else:
        raise ValueError(
            "Performance profile plotting only supported for atari/minatar/octax"
        )
    score_distributions, score_distributions_cis = rly.create_performance_profile(
        atari_200m_normalized_score_dict, atari_200m_thresholds
    )
    # Plot score distributions
    import seaborn as sns
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(ncols=1, figsize=(7, 3.8))

    if args.domain == "atari" and args.normalize:
        label = r"Human Normalized Score $(\tau)$"
    else:
        label = r"Score $(\tau)$"

    algo_styles = get_algorithm_linestyles()
    style_dict = {alg: algo_styles.get(alg, "-") for alg in sorted_exp_classes}

    plot_performance_profiles(
        score_distributions,
        atari_200m_thresholds,
        performance_profile_cis=score_distributions_cis,
        colors=color_dict,
        xlabel=label,
        ax=ax,
        linestyles=style_dict,
        use_non_linear_scaling=False,
    )
    ax.set_xlabel(label, fontsize=args.label_fontsize)
    ax.set_ylabel(
        ax.get_ylabel(), fontsize=args.label_fontsize
    )  # Keep current ylabel but set font
    ax.tick_params(axis="both", which="major", labelsize=args.tick_fontsize)

    # Create custom legend with matching colors and linestyles sorted by name
    sorted_algs_for_legend = sorted(valid_algorithms)
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=color_dict.get(alg, "#808080"),
            lw=2,
            linestyle=style_dict.get(alg, "-"),
            label=alg,
        )
        for alg in sorted_algs_for_legend
    ]
    ax.legend(handles=legend_elements, loc="best", fontsize=args.legend_fontsize)

    out_file = f"{args.domain}_rliable_performance_profile.pdf"
    plt.tight_layout()
    fig.savefig(out_file, dpi=150, bbox_inches="tight")
    print(f"Rliable aggregate plot saved to {out_file}")


def calculate_aggregate_metrics(
    data,
    target_step,
    filter_env_ids=None,
    filter_exp_classes=None,
    filter_seeds=None,
    normalize=False,
):
    """
    Calculate mean and median scores across all environments at a specific step.
    First averages scores across seeds within each environment, then aggregates
    across environments (matching the LaTeX table methodology).

    Args:
        data (dict): The data dictionary.
        target_step (int): The step parameter to check around.
        filter_env_ids (list): List of env_ids to include (None for all).
        filter_exp_classes (list): List of exp_classes to include (None for all).
        filter_seeds (list): List of seeds to include (None for all).
        normalize (bool): Whether to normalize scores.
    """
    # Load normalization data if needed
    normalization_data = None
    if normalize:
        normalization_data = load_normalization_data()

    # Structure: env_scores[exp_class][env_id] = [per_seed_scores]
    env_scores = {}

    # Identify all experiment classes we want to process
    target_exp_classes = set()
    for env_id in data:
        for exp_class in data[env_id]:
            if filter_exp_classes is None or exp_class in filter_exp_classes:
                target_exp_classes.add(exp_class)

    for exp_class in target_exp_classes:
        env_scores[exp_class] = {}

    # Iterate over environments
    for env_id, env_data in data.items():
        if filter_env_ids is not None and env_id not in filter_env_ids:
            continue

        # Skip if normalization is needed but data is missing
        if normalize and (
            normalization_data is None or env_id not in normalization_data
        ):
            continue

        for exp_class in target_exp_classes:
            if exp_class not in env_data:
                continue

            exp_runs = env_data[exp_class]
            seed_scores = []
            for seed, (step_vals, metric_vals) in exp_runs.items():
                if filter_seeds is not None and seed not in filter_seeds:
                    continue

                if len(step_vals) == 0:
                    continue

                # Find index for closest step
                idx = (np.abs(step_vals - target_step)).argmin()

                # Take avg of 100 return points around that step (50 before and 50 after)
                start_idx = max(0, idx - 50)
                end_idx = min(len(metric_vals), idx + 50)

                avg_ret = np.mean(metric_vals[start_idx:end_idx])
                if np.isnan(avg_ret):
                    continue

                # Apply normalization if needed
                if normalize and normalization_data and env_id in normalization_data:
                    rand_score = normalization_data[env_id][0]
                    human_score = normalization_data[env_id][-1]
                    avg_ret = (avg_ret - rand_score) / (human_score - rand_score)

                seed_scores.append(avg_ret)

            # Store per-environment mean (average across seeds first)
            if seed_scores:
                env_scores[exp_class][env_id] = np.mean(seed_scores)

    # Now aggregate across environments for each exp_class
    # Print results
    metric_label = "HNS" if normalize else "Score"
    print(f"\nAggregate Metrics at Step ~{target_step}")
    print(
        f"{'Experiment Class':<25} | {'Mean ' + metric_label:<12} | {'Median ' + metric_label:<12} | {'IQM ' + metric_label:<12} | {'N (Envs)':<8}"
    )
    print("-" * 77)

    results = {}
    for exp_class in sorted(env_scores.keys()):
        # Get per-environment mean scores (one value per environment)
        per_env_means = list(env_scores[exp_class].values())
        if per_env_means:
            mean_val = np.mean(per_env_means)
            median_val = np.median(per_env_means)

            # Calculate inter-quartile mean (IQM) across environments
            q1 = np.percentile(per_env_means, 25)
            q3 = np.percentile(per_env_means, 75)
            iqm_vals = [v for v in per_env_means if q1 <= v <= q3]
            iqm_val = np.mean(iqm_vals) if iqm_vals else np.nan

            count = len(per_env_means)
            results[exp_class] = {
                "mean": mean_val,
                "median": median_val,
                "iqm": iqm_val,
                "count": count,
            }

    # Sort results by IQM value
    sorted_results = sorted(results.items(), key=lambda x: x[1]["iqm"], reverse=True)

    for exp_class, metrics in sorted_results:
        print(
            f"{exp_class:<25} | {metrics['mean']:<12.4f} | {metrics['median']:<12.4f} | {metrics['iqm']:<12.4f} | {metrics['count']:<8}"
        )
    return results


def generate_aggregate_latex_table(
    data,
    target_step,
    args,
    bootstrap_reps=10000,
    confidence_level=0.95,
):
    """
    Generate a LaTeX table for aggregate metrics with confidence intervals.
    First averages scores across seeds within each environment, then aggregates
    across environments (matching the LaTeX table methodology).
    Highlights the best value in each column.

    Args:
        data (dict): The data dictionary.
        target_step (int): The step at which to evaluate metrics.
        args: Command line arguments containing normalize, exp_classes, filter_env_ids, domain.
        bootstrap_reps (int): Number of bootstrap samples for CI calculation.
        confidence_level (float): Confidence level for intervals (default 0.95).
    """
    normalize = args.normalize

    # Load normalization data if needed
    normalization_data = None
    if normalize:
        normalization_data = load_normalization_data()

    # Structure: env_scores[exp_class][env_id] = (per_env_mean, per_env_std)
    env_scores = {}

    # Identify all experiment classes we want to process
    target_exp_classes = set()
    for env_id in data:
        for exp_class in data[env_id]:
            if args.exp_classes is None or exp_class in args.exp_classes:
                target_exp_classes.add(exp_class)

    for exp_class in target_exp_classes:
        env_scores[exp_class] = {}

    # Iterate over environments
    for env_id, env_data in data.items():
        if args.filter_env_ids is not None and env_id not in args.filter_env_ids:
            continue

        # Skip if normalization is needed but data is missing
        if normalize and (
            normalization_data is None or env_id not in normalization_data
        ):
            continue

        for exp_class in target_exp_classes:
            if exp_class not in env_data:
                continue

            exp_runs = env_data[exp_class]
            seed_scores = []
            for seed, (step_vals, metric_vals) in exp_runs.items():
                if len(step_vals) == 0:
                    continue

                # Find index for closest step
                idx = (np.abs(step_vals - target_step)).argmin()

                # Take avg of 100 return points around that step (50 before and 50 after)
                start_idx = max(0, idx - 50)
                end_idx = min(len(metric_vals), idx + 50)

                avg_ret = np.mean(metric_vals[start_idx:end_idx])
                if np.isnan(avg_ret):
                    continue

                # Apply normalization if needed
                if normalize and normalization_data and env_id in normalization_data:
                    rand_score = normalization_data[env_id][0]
                    human_score = normalization_data[env_id][-1]
                    avg_ret = (avg_ret - rand_score) / (human_score - rand_score)

                seed_scores.append(avg_ret)

            # Store per-environment mean and std (average across seeds)
            if seed_scores:
                env_scores[exp_class][env_id] = (
                    np.mean(seed_scores),
                    np.std(seed_scores),
                )

    # Now collect per-environment means and stds for each exp_class
    scores = {}
    stds = {}
    for exp_class in target_exp_classes:
        scores[exp_class] = []
        stds[exp_class] = []
        for env_id in sorted(env_scores[exp_class].keys()):
            mean_val, std_val = env_scores[exp_class][env_id]
            scores[exp_class].append(mean_val)
            stds[exp_class].append(std_val)

    # Calculate metrics with standard deviations across per-environment means
    results = {}

    for exp_class in sorted(scores.keys()):
        vals = scores[exp_class]
        if not vals:
            continue

        vals_array = np.array(vals)

        # Mean: simple mean across environments
        mean_point = np.mean(vals_array)
        # Standard deviation across per-environment means
        mean_std = np.std(vals_array)

        # Median: median across environments
        median_point = np.median(vals_array)
        # Standard deviation across per-environment means
        median_std = np.std(vals_array)

        # IQM: inter-quartile mean across environments
        q1 = np.percentile(vals_array, 25)
        q3 = np.percentile(vals_array, 75)
        iqm_mask = (vals_array >= q1) & (vals_array <= q3)
        iqm_vals = vals_array[iqm_mask]

        if len(iqm_vals) > 0:
            iqm_point = np.mean(iqm_vals)
            # Standard deviation of IQM values
            iqm_std = np.std(iqm_vals)
        else:
            iqm_point = np.nan
            iqm_std = np.nan

        results[exp_class] = {
            "mean": (mean_point, mean_std),
            "median": (median_point, median_std),
            "iqm": (iqm_point, iqm_std),
            "count": len(vals),
        }

    if not results:
        print("No data available for aggregate LaTeX table.")
        return

    # Determine best values for each metric
    best_values = {}
    for metric in ["mean", "median", "iqm"]:
        max_val = max(
            results[exp_class][metric][0]
            for exp_class in results
            if not np.isnan(results[exp_class][metric][0])
        )
        best_values[metric] = max_val

    # Generate LaTeX table
    latex_lines = []
    norm_label = " (Human Normalized)" if normalize else ""
    latex_lines.append("\\begin{table}[h]")
    latex_lines.append("\\centering")
    latex_lines.append(
        f"\\caption{{Aggregate metrics at step {target_step}{norm_label}}}"
    )
    latex_lines.append("\\begin{tabular}{lccc}")
    latex_lines.append("\\hline")
    latex_lines.append("Algorithm & Mean & Median & IQM \\\\")
    latex_lines.append("\\hline")

    # Sort algorithms by IQM (descending)
    sorted_exp_classes = sorted(
        results.keys(), key=lambda k: results[k]["iqm"][0], reverse=True
    )

    for exp_class in sorted_exp_classes:
        row_parts = [exp_class]

        for metric in ["mean", "median", "iqm"]:
            point, std = results[exp_class][metric]

            # Format: point ± std
            cell_text = f"{point:.2f} $\\pm$ {std:.2f}"

            # Bold if best
            if not np.isnan(point) and abs(point - best_values[metric]) < 1e-6:
                cell_text = f"\\textbf{{{cell_text}}}"

            row_parts.append(cell_text)

        row = " & ".join(row_parts) + " \\\\"
        latex_lines.append(row)

    latex_lines.append("\\hline")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\label{tab:aggregate_results}")
    latex_lines.append("\\end{table}")

    # Write to file
    output_file = f"{args.domain}_aggregate_results_table.tex"
    with open(output_file, "w") as f:
        f.write("\n".join(latex_lines))

    print(f"\nAggregate LaTeX table saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot RL results from CSV files")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to directory containing CSV files",
        default=None,
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="minatar",
        help="Domain to filter (e.g., 'minatar', 'gymnax'). If not specified, plots all domains.",
    )
    parser.add_argument(
        "--exp-classes",
        type=str,
        nargs="+",
        default=None,
        help="List of experiment classes to include (default: all)",
    )
    parser.add_argument(
        "--use-frames",
        action="store_true",
        help="Plot x-axis in Frames (x4 steps) instead of Steps",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=51,
        help="Window size for Savitzky-Golay smoothing filter (must be odd, default: 51)",
    )
    parser.add_argument(
        "--aggregate-metrics",
        type=int,
        default=None,
        help="Calculate aggregate metrics at a specific step (optional)",
    )
    parser.add_argument(
        "--filter-env-ids",
        type=str,
        nargs="+",
        default=None,
        help="List of env_ids to include for aggregation (default: all)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Whether to normalize scores using human scores (for Atari).",
    )
    parser.add_argument(
        "--add-baseline",
        action="store_true",
        help="Add DQN baseline line to plots",
    )
    parser.add_argument(
        "--latex-table",
        action="store_true",
        help="Generate a LaTeX table",
    )
    parser.add_argument(
        "--latex-table-step",
        type=int,
        default=10000000,
        help="Step at which to evaluate for LaTeX table (required if --latex-table is set)",
    )
    parser.add_argument(
        "--latex-metric",
        type=str,
        default="mean",
        choices=["mean", "iqm"],
        help="Metric to report in LaTeX table: 'mean' or 'iqm' (default: mean)",
    )
    parser.add_argument(
        "--rliable-plot",
        action="store_true",
        help="Generate aggregate interval plots using rliable (requires rliable package)",
    )
    parser.add_argument(
        "--plot-metric",
        type=str,
        choices=["mean", "iqm"],
        default="iqm",
        help="Metric to plot for time series: 'mean' (default) or 'iqm' (inter-quartile mean)",
    )
    parser.add_argument(
        "--rliable-legend-bottom",
        action="store_true",
        help="Place legend at bottom instead of y-axis labels for rliable plots",
    )

    parser.add_argument(
        "--save-rliable-legend",
        action="store_true",
        help="Save rliable legend to a separate file",
    )
    parser.add_argument(
        "--tick-fontsize",
        type=str,
        default="large",
        help="Font size for plot ticks",
    )
    parser.add_argument(
        "--title-fontsize",
        type=str,
        default="xx-large",
        help="Font size for plot titles",
    )
    parser.add_argument(
        "--label-fontsize",
        type=str,
        default="x-large",
        help="Font size for axis labels",
    )
    parser.add_argument(
        "--legend-fontsize",
        type=str,
        default=13,
        help="Font size for legend text",
    )
    parser.add_argument(
        "--aggregate-latex-table",
        action="store_true",
        help="Generate LaTeX table for aggregate metrics with confidence intervals",
    )
    parser.add_argument(
        "--aggregate-latex-step",
        type=int,
        default=None,
        help="Step at which to evaluate for aggregate LaTeX table (defaults to --latex-table-step if not set)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of steps to plot on x-axis (optional)",
    )

    args = parser.parse_args()

    if args.normalize:
        assert (
            args.domain == "atari"
        ), "Normalization only supported for 'atari' domain."

    # Validate smoothing window is odd
    if args.smoothing_window % 2 == 0:
        print(
            f"Warning: smoothing_window must be odd. Adjusting {args.smoothing_window} to {args.smoothing_window + 1}"
        )
        args.smoothing_window += 1

    if not args.data_path:
        args.data_path = f"{args.domain}-data"

    # Load data
    data = load_csv_data(
        args.data_path, domain=args.domain, filter_exp_classes=args.exp_classes
    )

    if args.latex_table:
        if args.latex_table_step is None:
            print("Error: --latex-table-step is required when using --latex-table")
        else:
            generate_latex_table(data, args)

    if not data:
        print("No data loaded. Exiting.")
    else:
        # Plot results
        plot_results(data, args)

        # Calculate aggregate metrics if requested
        if args.aggregate_metrics is not None:
            calculate_aggregate_metrics(
                data,
                target_step=args.aggregate_metrics,
                filter_env_ids=args.filter_env_ids,
                filter_exp_classes=args.exp_classes,
                normalize=args.normalize,
            )

        # Generate aggregate LaTeX table if requested
        if args.aggregate_latex_table:
            target = args.aggregate_latex_step or args.latex_table_step
            if target is None:
                print(
                    "Error: --aggregate-latex-step or --latex-table-step is required when using --aggregate-latex-table"
                )
            else:
                generate_aggregate_latex_table(data, target, args)

        # Generate rliable aggregate plot if requested
        if args.rliable_plot:
            target = args.aggregate_metrics or args.latex_table_step
            if target is None:
                print(
                    "Provide --aggregate-metrics or --latex-table-step to use --rliable-plot"
                )
            else:
                plot_aggregate_with_rliable(data, args, target_step=target)
