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
