"""
Reusable UI components and blocks.
"""
import streamlit as st
import pandas as pd
from typing import Optional, List, Dict, Any

from src.ui.formatting import fmt_currency, fmt_hours, fmt_percent, fmt_rate


def kpi_strip(metrics: Dict[str, Any], 
              format_map: Optional[Dict[str, str]] = None):
    """
    Render horizontal strip of KPI cards.
    
    Args:
        metrics: Dict of {label: value}
        format_map: Dict of {label: format_type} where format_type is
                    'currency', 'hours', 'percent', 'rate', 'count'
    """
    if format_map is None:
        format_map = {}
    
    cols = st.columns(len(metrics))
    
    formatters = {
        "currency": fmt_currency,
        "hours": fmt_hours,
        "percent": fmt_percent,
        "rate": fmt_rate,
        "count": lambda x: f"{int(x):,}" if pd.notna(x) else "—",
    }
    
    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            fmt_type = format_map.get(label, "currency")
            formatter = formatters.get(fmt_type, str)
            
            formatted = formatter(value) if pd.notna(value) else "—"
            st.metric(label=label, value=formatted)


def risk_badge(status: str) -> str:
    """
    Return colored risk badge HTML.
    """
    colors = {
        "on_track": ("🟢", "#28a745"),
        "watch": ("🟡", "#ffc107"),
        "at_risk": ("🔴", "#dc3545"),
    }
    
    emoji, color = colors.get(status.lower(), ("⚪", "#6c757d"))
    return emoji


def status_indicator(value: float, 
                     good_threshold: float, 
                     bad_threshold: float,
                     invert: bool = False) -> str:
    """
    Return status indicator based on thresholds.
    
    Args:
        value: The value to check
        good_threshold: Above this is good (or bad if inverted)
        bad_threshold: Below this is bad (or good if inverted)
        invert: If True, lower is better
    """
    if pd.isna(value):
        return "⚪"
    
    if invert:
        if value <= good_threshold:
            return "🟢"
        elif value >= bad_threshold:
            return "🔴"
        return "🟡"
    else:
        if value >= good_threshold:
            return "🟢"
        elif value <= bad_threshold:
            return "🔴"
        return "🟡"


def metric_card_with_trend(label: str, 
                           value: str,
                           trend_value: Optional[float] = None,
                           trend_is_good: bool = True):
    """
    Render metric card with trend indicator.
    """
    delta = None
    delta_color = "normal"
    
    if trend_value is not None:
        if trend_value > 0:
            delta = f"+{trend_value:.1f}%"
            delta_color = "normal" if trend_is_good else "inverse"
        else:
            delta = f"{trend_value:.1f}%"
            delta_color = "inverse" if trend_is_good else "normal"
    
    st.metric(label=label, value=value, delta=delta, delta_color=delta_color)


def info_panel(title: str, items: List[Dict[str, str]]):
    """
    Render information panel with key-value pairs.
    """
    st.markdown(f"**{title}**")
    
    for item in items:
        label = item.get("label", "")
        value = item.get("value", "")
        st.markdown(f"- {label}: {value}")


def action_card(title: str,
                description: str,
                action_label: str,
                on_click: callable,
                key: str):
    """
    Render action card with button.
    """
    with st.container():
        st.markdown(f"**{title}**")
        st.caption(description)
        st.button(action_label, on_click=on_click, key=key)


def expandable_section(title: str, 
                       content_func: callable,
                       expanded: bool = False):
    """
    Render expandable section.
    """
    with st.expander(title, expanded=expanded):
        content_func()


def tab_container(tab_labels: List[str]) -> List:
    """
    Create tabs and return tab objects for content.
    """
    return st.tabs(tab_labels)


def progress_bar(label: str,
                 value: float,
                 max_value: float = 100,
                 color: str = "blue"):
    """
    Render labeled progress bar.
    """
    st.markdown(f"**{label}**")
    progress = min(value / max_value, 1.0) if max_value > 0 else 0
    st.progress(progress)
    st.caption(f"{value:.1f} / {max_value:.1f}")


def benchmark_badge(n_jobs: int, 
                    date_min: Optional[str] = None,
                    date_max: Optional[str] = None,
                    recency_weighted: bool = False):
    """
    Display benchmark sample metadata.
    """
    parts = [f"{n_jobs} jobs"]
    
    if date_min and date_max:
        parts.append(f"{date_min} - {date_max}")
    
    if recency_weighted:
        parts.append("recency-weighted")
    
    st.caption(" | ".join(parts))


def empty_state(message: str, 
                icon: str = "📭",
                action_label: Optional[str] = None,
                on_action: Optional[callable] = None,
                key: str = "empty_state"):
    """
    Render empty state with optional action.
    """
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown(f"### {icon}")
        st.markdown(f"**{message}**")
        
        if action_label and on_action:
            st.button(action_label, on_click=on_action, key=key)


def filter_chips(filters: Dict[str, Any]):
    """
    Display active filters as chips.
    """
    active = []
    
    for key, value in filters.items():
        if value is not None and value != "" and value != "All":
            if isinstance(value, bool):
                if value:
                    active.append(key.replace("_", " ").title())
            else:
                active.append(f"{key}: {value}")
    
    if active:
        chips = " | ".join([f"`{f}`" for f in active])
        st.caption(f"Active filters: {chips}")


def download_button(df: pd.DataFrame,
                    filename: str,
                    label: str = "Download CSV",
                    key: str = "download"):
    """
    Render download button for dataframe.
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
        key=key
    )
