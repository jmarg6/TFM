import os
import re
from typing import List
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import from our new configuration file
from utils.config import PrintMode, Algorithm

# Standard order to ensure charts always display algorithms consistently
ALGORITHM_ORDER = [
    "100%", 
    Algorithm.RANDOM_SEARCH.value, 
    Algorithm.LOCAL_SEARCH.value, 
    Algorithm.GENETIC.value, 
    Algorithm.MEMETIC.value, 
]

def plot_fitness_evolution(fitness_history: List[float], initial_percentage: int, algorithm_name: str, metric: str, model: str, output_dir: str):
    """
    Saves a simple line chart showing the evolution of fitness (e.g., during a Genetic Algorithm).
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(fitness_history, marker='o', linewidth=2, color='#1f77b4')
    
    ax.set_title(f'{metric.capitalize()} Evolution - {algorithm_name} - {model}\n(Initial Percentage: {initial_percentage}%)', fontsize=12, pad=10)
    ax.set_xlabel('Iteration / Generation', fontsize=11)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    safe_alg_name = algorithm_name.replace(" ", "_")
    output_path = os.path.join(output_dir, f'{model}-{safe_alg_name}-{initial_percentage}-{metric}.png')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_multiple_fitness_evolution(data: List[List[float]], labels: List[str], metric: str, title: str, filename: str, x_label: str = "Iteration"):
    """
    Compares the fitness evolution of multiple algorithms in a single line chart.
    """
    max_length = max((len(lst) for lst in data), default=50)
    max_length = max_length if max_length > 1 else 50

    # Pad shorter lists by repeating their last value (simulating early stopping)
    extended_data = [lst + [lst[-1]] * (max_length - len(lst)) for lst in data]

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, line_data in enumerate(extended_data):
        ax.plot(line_data, label=labels[i], linewidth=2)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(metric.capitalize(), fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def _plot_min_max_lines(df: pd.DataFrame, y_col: str, x_col: str, ax: plt.Axes):
    """
    Helper function to add minimum and maximum horizontal lines to boxplots.
    """
    # Enforce safe casting to string to prevent alignment mismatches
    df_temp = df.copy()
    df_temp[x_col] = df_temp[x_col].astype(str)
    
    grouped = df_temp.groupby(x_col)[y_col].agg(['min', 'max']).reset_index()
    xlabels = [str(tick.get_text()) for tick in ax.get_xticklabels()]
    xticks = ax.get_xticks()
    positions = dict(zip(xlabels, xticks))

    for _, row in grouped.iterrows():
        cat = str(row[x_col])
        if cat in positions:
            xpos = positions[cat]
            min_val, max_val = row['min'], row['max']
            ax.hlines(min_val, xpos - 0.2, xpos + 0.2, color='red', lw=1.5, linestyle='-')
            ax.hlines(max_val, xpos - 0.2, xpos + 0.2, color='blue', lw=1.5, linestyle='-')
            ax.text(xpos, min_val - 0.002, f'{min_val:.3f}', ha='center', va='top', fontsize=9, color='red', fontweight='bold')
            ax.text(xpos, max_val + 0.002, f'{max_val:.3f}', ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')


def plot_boxplot(df: pd.DataFrame, metric: str, filename: str | None, hue: str | None, title: str, x_axis: str, show_min_max=True):
    """
    Generates a boxplot to compare metric distributions (e.g., Accuracy vs Algorithm).
    """
    # Defensive lookup configuration to align matching headers with what Polars wrote
    metric_title = metric.capitalize() 
    if metric_title not in df.columns and metric.title() in df.columns:
        metric_title = metric.title()
        
    if metric_title not in df.columns or x_axis not in df.columns:
        print(f"Warning: Missing columns ({metric_title} or {x_axis}) in DataFrame. Boxplot skipped.")
        print(f"Available columns: {list(df.columns)}")
        return

    # Filter categories safely using string mappings
    categories = [str(c) for c in df[x_axis].dropna().unique()]
    try:
        order_x = sorted(categories, key=lambda x: float(x))
    except ValueError:
        order_x = [alg for alg in ALGORITHM_ORDER if str(alg) in categories]

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=x_axis, y=metric_title, hue=hue, order=order_x, ax=ax, palette="Pastel1")

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(x_axis, fontsize=11.5)
    ax.set_ylabel(metric_title, fontsize=11.5)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    if hue and hue in df.columns:
        ax.legend(title=hue, loc='best', fontsize=10, title_fontsize=11)

    if show_min_max:
        _plot_min_max_lines(df, metric_title, x_axis, ax)

    ax.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_barplot(df: pd.DataFrame, x: str, y: str, hue: str | None = None, order: list | None = None, 
                 palette: str = "Set2", errorbar: str | None = "sd", title: str = "", 
                 xlabel: str | None = None, ylabel: str | None = None, filename: str | None = None, 
                 show_values: bool = False):
    """
    Generates a generic barplot with standard error bars.
    """
    if x not in df.columns or y not in df.columns:
        return

    if order is None:
        try:
            order = sorted(df[x].dropna().unique(), key=lambda v: float(v))
        except (ValueError, TypeError):
            order = list(df[x].dropna().unique())

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x=x, y=y, hue=hue, order=order, palette=palette, errorbar=errorbar, ax=ax)

    if show_values:
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", fontsize=10, fontweight='bold', label_type="edge", padding=2)

    if title:
        ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel(xlabel or x, fontsize=11.5)
    ax.set_ylabel(ylabel or y, fontsize=11.5)
    
    ax.tick_params(axis='both', labelsize=11)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_initial_vs_final_scatter(df: pd.DataFrame, filename: str | None = None, 
                                  title: str = "Initial vs Final Percentage (Colored by Accuracy)"):
    """
    Generates a scatter plot critical for Instance Selection: visualizes dataset reduction.
    """
    # Defensive lookup matching column variations
    x_col = "Initial Percentage" if "Initial Percentage" in df.columns else "Initial_Percentage"
    y_col = "Final Percentage" if "Final Percentage" in df.columns else "Final_Percentage"
    target_metric = "Accuracy" if "Accuracy" in df.columns else "Accuracy"

    if not {x_col, y_col, target_metric}.issubset(df.columns):
        print("Warning: Missing required columns for scatter plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df, x=x_col, y=y_col,
        size=target_metric, hue=target_metric, sizes=(50, 300),
        palette="coolwarm", alpha=0.8, edgecolor="black", linewidth=0.5, ax=ax
    )

    for _, row in df.iterrows():
        pi, pf = row[x_col], row[y_col]
        ax.vlines(x=pi, ymin=0, ymax=pf, color="dimgray", alpha=0.6, linestyle="--", linewidth=1.2)
        ax.hlines(y=pf, xmin=0, xmax=pi, color="dimgray", alpha=0.6, linestyle="--", linewidth=1.2)

    ax.set_xticks([0.1, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0.1, 0.25, 0.5, 0.75, 1.0])
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
        
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)

    ax.set_title(title, fontsize=13, pad=12)
    ax.set_xlabel("Initial Percentage (Original Dataset)", fontsize=11.5)
    ax.set_ylabel("Final Percentage (After Algorithm)", fontsize=11.5)

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)