import json
import uuid
from collections import OrderedDict
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from pydantic import BaseModel, Field

# Ephemeral chart data — consumed by the /chat endpoint immediately after a response.
_pending_charts: dict = {}
# Persistent chart data — retained for the /history endpoint so charts survive refresh.
# Capped at 500 entries (oldest evicted first) to bound memory usage.
_CHART_STORE_MAX = 500
_chart_store: OrderedDict = OrderedDict()


class ChartQuery(BaseModel):
    tickers: str = Field(
        ...,
        description="Comma-separated ticker symbols, e.g. 'AAPL' or 'AAPL,MSFT,GOOGL'",
    )
    chart_type: Literal["price_history", "comparison", "metrics"] = Field(
        ...,
        description="Chart type: 'price_history' (single stock close price), "
                    "'comparison' (multi-stock % change), or 'metrics' (financial metrics bar chart)",
    )
    period: Literal["7d", "30d", "90d", "180d", "1y"] = Field(
        default="30d",
        description="Time period: '7d', '30d', '90d', '180d', or '1y'",
    )


def _price_history_chart(ticker: str, period: str) -> dict:
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No price data found for {ticker}")

    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=close.index.tolist(),
        y=close.values.tolist(),
        mode="lines",
        name=ticker,
        line=dict(color="#2196F3", width=2),
    ))
    fig.update_layout(
        title=f"{ticker} Price History ({period})",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_white",
    )
    return json.loads(fig.to_json())


def _comparison_chart(tickers: list, period: str) -> dict:
    data = yf.download(tickers, period=period, progress=False)
    if data.empty:
        raise ValueError(f"No price data found for {', '.join(tickers)}")

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data[["Close"]]
        close.columns = tickers

    fig = go.Figure()
    for ticker in close.columns:
        series = close[ticker].dropna()
        if series.empty:
            continue
        normalized = (series / series.iloc[0] - 1) * 100
        fig.add_trace(go.Scatter(
            x=normalized.index.tolist(),
            y=normalized.values.tolist(),
            mode="lines",
            name=ticker,
        ))
    if not fig.data:
        raise ValueError(f"No data available for any of the requested tickers: {', '.join(tickers)}")
    fig.update_layout(
        title=f"Price Comparison ({period})",
        xaxis_title="Date",
        yaxis_title="% Change from Start",
        template="plotly_white",
    )
    return json.loads(fig.to_json())


def _metrics_chart(ticker: str) -> dict:
    info = yf.Ticker(ticker).info or {}

    metrics = {
        "P/E Ratio": info.get("trailingPE"),
        "EPS": info.get("trailingEps"),
        "Beta": info.get("beta"),
        "Dividend Yield (%)": (info.get("dividendYield") or 0) * 100,
        "Debt/Equity": info.get("debtToEquity"),
    }

    labels = [k for k, v in metrics.items() if v is not None]
    values = [metrics[k] for k in labels]

    if not labels:
        raise ValueError(f"No financial metrics available for {ticker}")

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color="#2196F3",
    ))
    fig.update_layout(
        title=f"{ticker} Financial Metrics",
        template="plotly_white",
        xaxis_title="Value",
    )
    return json.loads(fig.to_json())


def generate_chart(tickers: str, chart_type: str, period: str = "30d") -> str:
    """
    Generate a Plotly chart for one or more stock tickers.

    Returns JSON string {"plot_data": <plotly figure dict>, "description": "<summary>"}
    or a plain error string on failure (so the agent can respond gracefully).
    """
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    if not ticker_list:
        return "No valid tickers provided."

    try:
        if chart_type == "price_history":
            plot_data = _price_history_chart(ticker_list[0], period)
            description = f"Price history for {ticker_list[0]} over {period}."
        elif chart_type == "comparison":
            plot_data = _comparison_chart(ticker_list, period)
            description = f"Normalised price comparison for {', '.join(ticker_list)} over {period}."
        elif chart_type == "metrics":
            plot_data = _metrics_chart(ticker_list[0])
            description = f"Financial metrics for {ticker_list[0]}."
        else:
            return (
                f"Unknown chart_type '{chart_type}'. "
                "Use 'price_history', 'comparison', or 'metrics'."
            )

        chart_id = uuid.uuid4().hex[:8]
        _pending_charts[chart_id] = plot_data
        _chart_store[chart_id] = plot_data
        if len(_chart_store) > _CHART_STORE_MAX:
            _chart_store.popitem(last=False)
        return f"Chart generated [chart:{chart_id}]: {description} It will be displayed to the user automatically."
    except Exception as e:
        return f"Could not generate chart: {str(e)}"
