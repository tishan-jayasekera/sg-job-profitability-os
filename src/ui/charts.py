"""
Standard chart wrappers using Plotly.
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any


# =============================================================================
# CHART THEME
# =============================================================================

CHART_COLORS = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#28a745",
    "warning": "#ffc107",
    "danger": "#dc3545",
    "neutral": "#6c757d",
    "light": "#f8f9fa",
}

CHART_TEMPLATE = "plotly_white"

DEFAULT_LAYOUT = {
    "template": CHART_TEMPLATE,
    "font": {"family": "Arial, sans-serif", "size": 12},
    "margin": {"l": 50, "r": 30, "t": 40, "b": 50},
    "hoverlabel": {"bgcolor": "white"},
}


def apply_layout(fig: go.Figure, **kwargs) -> go.Figure:
    """Apply standard layout to figure."""
    layout = {**DEFAULT_LAYOUT, **kwargs}
    fig.update_layout(**layout)
    return fig


# =============================================================================
# KPI CHARTS
# =============================================================================

def kpi_gauge(value: float, target: Optional[float] = None, 
              title: str = "", suffix: str = "%",
              color: str = "primary") -> go.Figure:
    """
    Create a gauge chart for KPI display.
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title},
        number={"suffix": suffix},
        gauge={
            "axis": {"range": [0, 100] if suffix == "%" else [None, None]},
            "bar": {"color": CHART_COLORS.get(color, color)},
            "threshold": {
                "line": {"color": CHART_COLORS["danger"], "width": 2},
                "value": target,
            } if target else None,
        },
    ))
    
    return apply_layout(fig, height=200)


# =============================================================================
# BAR CHARTS
# =============================================================================

def horizontal_bar(df: pd.DataFrame, x: str, y: str, 
                   title: str = "", color: Optional[str] = None,
                   text: Optional[str] = None) -> go.Figure:
    """
    Create horizontal bar chart.
    """
    fig = px.bar(
        df, x=x, y=y, orientation="h",
        title=title,
        color=color,
        text=text,
    )
    
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    
    return apply_layout(fig)


def grouped_bar(df: pd.DataFrame, x: str, y: List[str],
                title: str = "", barmode: str = "group") -> go.Figure:
    """
    Create grouped or stacked bar chart.
    """
    fig = go.Figure()
    
    colors = list(CHART_COLORS.values())
    
    for i, col in enumerate(y):
        fig.add_trace(go.Bar(
            name=col,
            x=df[x],
            y=df[col],
            marker_color=colors[i % len(colors)],
        ))
    
    fig.update_layout(barmode=barmode, title=title)
    
    return apply_layout(fig)


def waterfall_chart(df: pd.DataFrame, 
                    labels: str, values: str,
                    title: str = "") -> go.Figure:
    """
    Create waterfall chart for margin bridge or variance decomposition.
    
    df should have columns:
    - labels: category names
    - values: contribution values (can be negative)
    - measure (optional): 'relative', 'total', or 'absolute'
    """
    measures = df.get("measure", ["relative"] * len(df))
    
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measures,
        x=df[labels],
        y=df[values],
        textposition="outside",
        text=[f"${v:,.0f}" for v in df[values]],
        connector={"line": {"color": CHART_COLORS["neutral"]}},
        increasing={"marker": {"color": CHART_COLORS["success"]}},
        decreasing={"marker": {"color": CHART_COLORS["danger"]}},
        totals={"marker": {"color": CHART_COLORS["primary"]}},
    ))
    
    fig.update_layout(title=title)
    
    return apply_layout(fig)


# =============================================================================
# SCATTER PLOTS
# =============================================================================

def scatter_plot(df: pd.DataFrame, x: str, y: str,
                 size: Optional[str] = None, color: Optional[str] = None,
                 hover_name: Optional[str] = None,
                 hover_data: Optional[List[str]] = None,
                 title: str = "",
                 x_title: str = "", y_title: str = "") -> go.Figure:
    """
    Create scatter plot with optional size and color encoding.
    """
    fig = px.scatter(
        df, x=x, y=y,
        size=size, color=color,
        hover_name=hover_name,
        hover_data=hover_data,
        title=title,
    )
    
    fig.update_layout(
        xaxis_title=x_title or x,
        yaxis_title=y_title or y,
    )
    
    # Add reference line if x and y are comparable
    if df[x].min() > 0 and df[y].min() > 0:
        min_val = min(df[x].min(), df[y].min())
        max_val = max(df[x].max(), df[y].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"dash": "dash", "color": CHART_COLORS["neutral"]},
            name="x=y",
            showlegend=False,
        ))
    
    return apply_layout(fig)


def client_quadrant_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    size_col: str,
    quadrant_col: str,
    median_x: float,
    median_y: float,
    title: str = "Client Portfolio Quadrants",
    x_title: str = "Revenue",
    y_title: str = "Profit",
) -> go.Figure:
    """
    Scatter with quadrant lines and labels.
    """
    if len(df) == 0:
        return apply_layout(go.Figure(), title=title)

    df_plot = df.copy()
    if "margin" in df_plot.columns:
        df_plot["margin_sign"] = np.where(df_plot["margin"] >= 0, "Accretive", "Erosive")
    else:
        df_plot["margin_sign"] = "Accretive"

    fig = px.scatter(
        df_plot,
        x=x_col,
        y=y_col,
        size=size_col,
        color="margin_sign",
        symbol=quadrant_col,
        hover_name="client",
        hover_data=["revenue", "margin", "margin_pct", "hours", "realised_rate"],
        title=title,
        color_discrete_map={
            "Accretive": "#2ca02c",
            "Erosive": "#d62728",
        },
    )

    fig.add_vline(x=median_x, line_dash="dash", line_color=CHART_COLORS["neutral"])
    fig.add_hline(y=median_y, line_dash="dash", line_color=CHART_COLORS["neutral"])
    fig.add_hline(y=0, line_dash="dot", line_color=CHART_COLORS["danger"])
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.98,
        text="Margin Accretive",
        showarrow=False,
        font=dict(color="#2ca02c"),
    )
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.02,
        y=0.02,
        text="Margin Erosive",
        showarrow=False,
        font=dict(color="#d62728"),
    )
    fig.update_layout(xaxis_title=x_title, yaxis_title=y_title, dragmode="select")

    return apply_layout(fig)


def render_risk_matrix(df: pd.DataFrame,
                       x: str,
                       y: str,
                       size: str,
                       color: str,
                       hover_name: str,
                       hover_data: List[str],
                       title: str = "Risk Matrix",
                       custom_data: Optional[List[str]] = None) -> go.Figure:
    """
    Render risk matrix scatter with shaded risk zones.
    """
    fig = px.scatter(
        df,
        x=x,
        y=y,
        size=size,
        color=color,
        hover_name=hover_name,
        hover_data=hover_data,
        custom_data=custom_data,
        color_continuous_scale=["#2ca02c", "#ffbf00", "#d62728"],
        range_color=(0, 1),
        title=title,
    )
    fig.add_vline(x=0, line_dash="dash", line_color=CHART_COLORS["danger"])
    fig.add_vline(x=2, line_dash="dot", line_color=CHART_COLORS["warning"])
    fig.update_layout(xaxis_title="Time Buffer (weeks)", yaxis_title="Remaining Hours")
    return apply_layout(fig)


def render_task_stacked_bar(df: pd.DataFrame,
                            task_col: str,
                            expected_col: str,
                            actual_col: str,
                            remaining_col: str,
                            title: str = "Task Shape vs Reality") -> go.Figure:
    """
    Stacked bar chart for task benchmark vs actual vs remaining.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df[task_col], y=df[expected_col], name="Benchmark (Expected)", marker_color="#9ecae1"))
    fig.add_trace(go.Bar(x=df[task_col], y=df[actual_col], name="Actual", marker_color="#3182bd"))
    fig.add_trace(go.Bar(x=df[task_col], y=df[remaining_col], name="Remaining", marker_color="#ff7f0e"))
    fig.update_layout(barmode="stack", title=title, xaxis_title="Task", yaxis_title="Hours")
    return apply_layout(fig)


def render_bottleneck_heatmap(df: pd.DataFrame,
                              job_col: str,
                              task_col: str,
                              value_col: str,
                              status_col: str,
                              title: str = "Bottleneck Heatmap") -> go.Figure:
    """
    Heatmap of job-task bottleneck status with remaining hours annotations.
    """
    pivot = df.pivot_table(index=job_col, columns=task_col, values=value_col, aggfunc="sum").fillna(0)
    status = df.pivot_table(index=job_col, columns=task_col, values=status_col, aggfunc="first")
    z = pivot.values
    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=pivot.columns,
        y=pivot.index,
        colorscale="YlOrRd",
        colorbar_title="Remaining Hrs",
    ))
    fig.update_layout(title=title, xaxis_title="Task", yaxis_title="Job")
    return apply_layout(fig)


def rate_scatter(df: pd.DataFrame, 
                 group_col: str,
                 revenue_col: str = "revenue",
                 quote_rate_col: str = "quote_rate",
                 realised_rate_col: str = "realised_rate",
                 title: str = "Quote Rate vs Realised Rate") -> go.Figure:
    """
    Create rate capture scatter: X=quote rate, Y=realised rate, size=revenue.
    """
    fig = px.scatter(
        df, 
        x=quote_rate_col, 
        y=realised_rate_col,
        size=revenue_col,
        hover_name=group_col,
        hover_data=[revenue_col, quote_rate_col, realised_rate_col],
        title=title,
    )
    
    # Add x=y reference line
    rate_min = min(df[quote_rate_col].min(), df[realised_rate_col].min())
    rate_max = max(df[quote_rate_col].max(), df[realised_rate_col].max())
    
    fig.add_trace(go.Scatter(
        x=[rate_min, rate_max],
        y=[rate_min, rate_max],
        mode="lines",
        line={"dash": "dash", "color": CHART_COLORS["neutral"]},
        name="Quote = Realised",
        showlegend=True,
    ))
    
    fig.update_layout(
        xaxis_title="Quote Rate ($/hr)",
        yaxis_title="Realised Rate ($/hr)",
    )
    
    return apply_layout(fig)


# =============================================================================
# DISTRIBUTION CHARTS
# =============================================================================

def histogram(df: pd.DataFrame, x: str, 
              title: str = "", nbins: int = 30,
              color: str = "primary") -> go.Figure:
    """
    Create histogram.
    """
    fig = px.histogram(
        df, x=x,
        title=title,
        nbins=nbins,
    )
    
    fig.update_traces(marker_color=CHART_COLORS.get(color, color))
    
    return apply_layout(fig)


def box_plot(df: pd.DataFrame, x: str, y: str,
             title: str = "") -> go.Figure:
    """
    Create box plot.
    """
    fig = px.box(
        df, x=x, y=y,
        title=title,
    )
    
    return apply_layout(fig)


# =============================================================================
# TIME SERIES
# =============================================================================

def time_series(df: pd.DataFrame, x: str, y: str,
                color: Optional[str] = None,
                title: str = "", 
                y_title: str = "") -> go.Figure:
    """
    Create time series line chart.
    """
    fig = px.line(
        df, x=x, y=y,
        color=color,
        title=title,
        markers=True,
    )
    
    fig.update_layout(
        xaxis_title="",
        yaxis_title=y_title or y,
    )
    
    return apply_layout(fig)


def cumulative_profit_line(df: pd.DataFrame, x: str, y: str,
                           title: str = "Cumulative Profit") -> go.Figure:
    """
    Cumulative profit line chart.
    """
    fig = px.line(df, x=x, y=y, title=title, markers=True)
    fig.update_layout(xaxis_title="", yaxis_title="Cumulative Profit")
    return apply_layout(fig)


def multi_time_series(df: pd.DataFrame, x: str, y_cols: List[str],
                      title: str = "") -> go.Figure:
    """
    Create time series with multiple metrics.
    """
    fig = go.Figure()
    
    colors = list(CHART_COLORS.values())
    
    for i, col in enumerate(y_cols):
        fig.add_trace(go.Scatter(
            x=df[x],
            y=df[col],
            name=col,
            mode="lines+markers",
            line={"color": colors[i % len(colors)]},
        ))
    
    fig.update_layout(title=title, xaxis_title="")
    
    return apply_layout(fig)


# =============================================================================
# CAPACITY CHARTS
# =============================================================================

def capacity_bar(supply: float, load: float, headroom: float,
                 title: str = "Capacity Overview") -> go.Figure:
    """
    Create stacked capacity bar showing supply/load/headroom.
    """
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name="Load",
        x=["Capacity"],
        y=[load],
        marker_color=CHART_COLORS["primary"],
    ))
    
    fig.add_trace(go.Bar(
        name="Headroom",
        x=["Capacity"],
        y=[headroom],
        marker_color=CHART_COLORS["success"],
    ))
    
    # Add supply line
    fig.add_hline(
        y=supply,
        line_dash="dash",
        line_color=CHART_COLORS["neutral"],
        annotation_text=f"Supply: {supply:,.0f}",
    )
    
    fig.update_layout(
        barmode="stack",
        title=title,
        yaxis_title="Hours",
    )
    
    return apply_layout(fig, height=300)


# =============================================================================
# QUADRANT CHART
# =============================================================================

def quadrant_scatter(df: pd.DataFrame, x: str, y: str,
                     hover_name: str,
                     x_median: Optional[float] = None,
                     y_median: Optional[float] = None,
                     title: str = "",
                     x_title: str = "", y_title: str = "",
                     size: Optional[str] = None) -> go.Figure:
    """
    Create quadrant scatter with median reference lines.
    """
    if x_median is None:
        x_median = df[x].median()
    if y_median is None:
        y_median = df[y].median()
    
    fig = px.scatter(
        df, x=x, y=y,
        hover_name=hover_name,
        size=size,
        title=title,
    )
    
    # Add quadrant lines
    fig.add_vline(x=x_median, line_dash="dash", line_color=CHART_COLORS["neutral"])
    fig.add_hline(y=y_median, line_dash="dash", line_color=CHART_COLORS["neutral"])
    
    # Add quadrant labels
    x_range = df[x].max() - df[x].min()
    y_range = df[y].max() - df[y].min()
    
    fig.add_annotation(
        x=x_median + x_range * 0.3, y=y_median + y_range * 0.3,
        text="High Value / High Effort",
        showarrow=False, font={"size": 10, "color": CHART_COLORS["neutral"]}
    )
    fig.add_annotation(
        x=x_median - x_range * 0.3, y=y_median + y_range * 0.3,
        text="High Value / Low Effort",
        showarrow=False, font={"size": 10, "color": CHART_COLORS["success"]}
    )
    
    fig.update_layout(
        xaxis_title=x_title or x,
        yaxis_title=y_title or y,
    )
    
    return apply_layout(fig)


# =============================================================================
# RISK MATRIX & BOTTLENECK CHARTS (NEW FOR FORECAST PAGE)
# =============================================================================

def risk_matrix(job_level: pd.DataFrame, on_click_job_callback=None) -> go.Figure:
    """
    Create risk heat-map scatter plot.
    
    X-axis: Time buffer (weeks until due - weeks to complete)
    Y-axis: Remaining work (hours)
    Bubble size: Team velocity (hours/week)
    Color: Risk score (0=green, 1=red)
    
    Args:
        job_level: DataFrame with columns: job_no, due_weeks, job_eta_weeks, 
                   remaining_task_hours (or similar), team_velocity (optional), risk_score
        on_click_job_callback: Optional callback function(job_no) on bubble click
        
    Returns:
        Plotly Figure
    """
    df = job_level.copy()
    df = df.dropna(subset=["due_weeks", "job_eta_weeks"])
    
    # Compute time buffer
    df["time_buffer"] = df["due_weeks"] - df["job_eta_weeks"]
    
    # Estimate remaining hours if not present
    if "remaining_task_hours" not in df.columns:
        df["remaining_task_hours"] = 100  # Fallback
    
    # Estimate velocity if not present
    if "team_velocity" not in df.columns:
        df["team_velocity"] = 38  # Default FTE 38 hrs/week
    
    # Ensure risk_score exists
    if "risk_score" not in df.columns:
        from ..modeling.forecast import compute_risk_score
        df["risk_score"] = df.apply(
            lambda row: compute_risk_score(row["due_weeks"], row["job_eta_weeks"]),
            axis=1
        )
    
    # Color mapping: 0=green, 1=red
    colors = df["risk_score"].fillna(0.5)
    
    # Status labels
    df["status_label"] = df.apply(
        lambda row: "On-Track" if row["risk_score"] < 0.2 else (
            "Caution" if row["risk_score"] < 0.5 else (
                "At-Risk" if row["risk_score"] < 0.8 else "Critical"
            )
        ) if pd.notna(row["risk_score"]) else "Unknown",
        axis=1
    )
    
    fig = go.Figure()
    
    # Add scatter trace
    fig.add_trace(go.Scatter(
        x=df["time_buffer"],
        y=df["remaining_task_hours"],
        mode="markers",
        marker=dict(
            size=df["team_velocity"].fillna(38) / 2,
            color=colors,
            colorscale="RdYlGn_r",
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=dict(
                title="Risk Score",
                tickvals=[0, 0.25, 0.5, 0.75, 1.0],
                ticktext=["On-Track", "Caution", "Yellow", "At-Risk", "Critical"],
            ),
            line=dict(width=1, color="white"),
        ),
        text=[f"Job: {j}<br>Status: {s}<br>Buffer: {b:.1f}w<br>Remaining: {r:.0f}h<br>Risk: {rs:.2f}" 
              for j, s, b, r, rs in zip(df["job_no"], df["status_label"], 
                                        df["time_buffer"], df["remaining_task_hours"], 
                                        df["risk_score"])],
        hovertemplate="%{text}<extra></extra>",
        customdata=df["job_no"],
        name="Jobs",
    ))
    
    # Add reference lines
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Due Now", annotation_position="top right")
    fig.add_vline(x=2, line_dash="dot", line_color="orange", annotation_text="2-Week Warning", annotation_position="top right")
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
    
    # Add shaded risk zones
    fig.add_shape(type="rect", x0=-100, x1=0, y0=0, y1=df["remaining_task_hours"].max() * 1.1,
                  fillcolor="red", opacity=0.05, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=0, x1=2, y0=0, y1=df["remaining_task_hours"].max() * 1.1,
                  fillcolor="orange", opacity=0.05, line_width=0, layer="below")
    fig.add_shape(type="rect", x0=2, x1=100, y0=0, y1=df["remaining_task_hours"].max() * 1.1,
                  fillcolor="green", opacity=0.05, line_width=0, layer="below")
    
    fig.update_layout(
        title="Job Risk Matrix (Time Buffer vs. Remaining Work)",
        xaxis_title="Time Buffer (weeks)",
        yaxis_title="Remaining Work (hours)",
        height=500,
        hovermode="closest",
    )
    
    return apply_layout(fig)


def task_stacked_bar(task_data: pd.DataFrame) -> go.Figure:
    """
    Create stacked bar chart showing task decomposition (expected, actual, remaining).
    
    Args:
        task_data: DataFrame with columns: task_name, expected_task_hours, 
                   actual_task_hours, remaining_task_hours
                   
    Returns:
        Plotly Figure
    """
    df = task_data.copy()
    df = df.fillna(0)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df["task_name"],
        y=df["expected_task_hours"],
        name="Benchmark (Expected)",
        marker_color="#b3d9ff",
    ))
    
    fig.add_trace(go.Bar(
        x=df["task_name"],
        y=df["actual_task_hours"],
        name="Actual",
        marker_color="#1f77b4",
    ))
    
    fig.add_trace(go.Bar(
        x=df["task_name"],
        y=df["remaining_task_hours"],
        name="Forecast Remaining",
        marker_color="#ff7f0e",
    ))
    
    fig.update_layout(
        barmode="stack",
        title="Task Shape vs. Reality (Benchmark → Actual → Remaining)",
        xaxis_title="Task",
        yaxis_title="Hours",
        height=320,
        hovermode="x unified",
    )
    
    return apply_layout(fig)


def bottleneck_heatmap(job_task_matrix: pd.DataFrame) -> go.Figure:
    """
    Create heatmap showing bottleneck status by job and task.
    
    Args:
        job_task_matrix: Pivoted DataFrame with jobs as rows, tasks as columns,
                        values = status (0=green, 1=yellow, 2=red) or remaining hours
                        
    Returns:
        Plotly Figure
    """
    fig = go.Figure(data=go.Heatmap(
        z=job_task_matrix.values,
        x=job_task_matrix.columns,
        y=job_task_matrix.index,
        colorscale="RdYlGn_r",
        hovertemplate="Job: %{y}<br>Task: %{x}<br>Value: %{z:.0f}<extra></extra>",
    ))
    
    fig.update_layout(
        title="Bottleneck Heatmap (Job × Task)",
        xaxis_title="Task",
        yaxis_title="Job",
        height=400,
    )
    
    return apply_layout(fig)
