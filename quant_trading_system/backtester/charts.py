"""
AlphaDesk — Backtest Visualization
Generates performance charts as HTML (standalone, no server needed).
"""

import json
from datetime import datetime
from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_backtest_report_html(result, output_path: str = "backtest_report.html"):
    """
    Generate a comprehensive HTML backtest report with interactive charts.
    Uses Chart.js via CDN — works offline after first load.
    """

    # Prepare data
    equity_dates = [e[0].strftime("%Y-%m-%d") if isinstance(e[0], (datetime, pd.Timestamp)) else str(e[0])
                    for e in result.equity_curve]
    equity_values = [round(e[1], 2) for e in result.equity_curve]

    # Drawdown series
    peak = result.initial_capital
    drawdowns = []
    for _, eq in result.equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / peak * 100
        drawdowns.append(round(dd, 2))

    # Monthly returns
    trades_df = result.to_dataframe()
    monthly_pnl = {}
    if not trades_df.empty and 'exit_date' in trades_df.columns:
        for _, trade in trades_df.iterrows():
            month_key = trade['exit_date'].strftime("%Y-%m") if isinstance(trade['exit_date'], (datetime, pd.Timestamp)) else str(trade['exit_date'])[:7]
            monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + trade['pnl']

    monthly_labels = list(monthly_pnl.keys())
    monthly_values = [round(v, 2) for v in monthly_pnl.values()]
    monthly_colors = ['rgba(34,197,94,0.7)' if v >= 0 else 'rgba(239,68,68,0.7)' for v in monthly_values]

    # Win/Loss distribution
    pnl_pcts = [round(t.pnl_pct * 100, 2) for t in result.trades] if result.trades else []

    # Exit reason breakdown
    exit_labels = list(result.exit_reasons.keys())
    exit_counts = [result.exit_reasons[k]["count"] for k in exit_labels]
    exit_pnls = [round(result.exit_reasons[k]["pnl"], 2) for k in exit_labels]

    # Strategy metrics for summary cards
    years = len(result.daily_returns) / 252 if len(result.daily_returns) > 0 else 1
    annual_return = ((1 + result.total_return) ** (1 / years) - 1) if years > 0 else 0

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlphaDesk Backtest Report</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
        .header {{ text-align: center; margin-bottom: 32px; }}
        .header h1 {{ font-size: 28px; color: #f8fafc; margin-bottom: 8px; }}
        .header .subtitle {{ color: #94a3b8; font-size: 14px; }}

        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 20px; text-align: center; }}
        .card .value {{ font-size: 24px; font-weight: 700; margin: 8px 0; }}
        .card .label {{ font-size: 12px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; }}
        .positive {{ color: #34d399; }}
        .negative {{ color: #f87171; }}
        .neutral {{ color: #60a5fa; }}

        .chart-container {{ background: #1e293b; border-radius: 12px; padding: 24px; margin-bottom: 24px; }}
        .chart-container h2 {{ font-size: 16px; color: #cbd5e1; margin-bottom: 16px; }}
        .chart-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
        @media (max-width: 768px) {{ .chart-row {{ grid-template-columns: 1fr; }} }}

        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; }}
        th {{ text-align: left; padding: 8px 12px; border-bottom: 2px solid #334155; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #1e293b; font-size: 14px; }}
        tr:hover {{ background: #1e293b; }}

        .footer {{ text-align: center; color: #475569; font-size: 12px; margin-top: 40px; padding-top: 20px; border-top: 1px solid #1e293b; }}
    </style>
</head>
<body>

<div class="header">
    <h1>AlphaDesk — Backtest Report</h1>
    <div class="subtitle">Generated {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}</div>
</div>

<!-- KPI Cards -->
<div class="cards">
    <div class="card">
        <div class="label">Total Return</div>
        <div class="value {'positive' if result.total_return >= 0 else 'negative'}">{result.total_return:+.2%}</div>
    </div>
    <div class="card">
        <div class="label">Annual Return</div>
        <div class="value {'positive' if annual_return >= 0 else 'negative'}">{annual_return:+.2%}</div>
    </div>
    <div class="card">
        <div class="label">Sharpe Ratio</div>
        <div class="value neutral">{result.sharpe_ratio:.2f}</div>
    </div>
    <div class="card">
        <div class="label">Max Drawdown</div>
        <div class="value negative">-{result.max_drawdown:.2%}</div>
    </div>
    <div class="card">
        <div class="label">Win Rate</div>
        <div class="value {'positive' if result.win_rate >= 0.5 else 'negative'}">{result.win_rate:.1%}</div>
    </div>
    <div class="card">
        <div class="label">Profit Factor</div>
        <div class="value {'positive' if result.profit_factor >= 1 else 'negative'}">{result.profit_factor:.2f}</div>
    </div>
    <div class="card">
        <div class="label">Total Trades</div>
        <div class="value neutral">{result.num_trades}</div>
    </div>
    <div class="card">
        <div class="label">Total P&L</div>
        <div class="value {'positive' if result.total_pnl >= 0 else 'negative'}">${result.total_pnl:+,.0f}</div>
    </div>
</div>

<!-- Equity Curve -->
<div class="chart-container">
    <h2>Equity Curve</h2>
    <canvas id="equityChart" height="100"></canvas>
</div>

<!-- Drawdown -->
<div class="chart-container">
    <h2>Drawdown</h2>
    <canvas id="drawdownChart" height="60"></canvas>
</div>

<!-- Monthly Returns + Distribution -->
<div class="chart-row">
    <div class="chart-container">
        <h2>Monthly P&L</h2>
        <canvas id="monthlyChart" height="120"></canvas>
    </div>
    <div class="chart-container">
        <h2>Return Distribution</h2>
        <canvas id="distChart" height="120"></canvas>
    </div>
</div>

<!-- Exit Reason breakdown -->
<div class="chart-row">
    <div class="chart-container">
        <h2>Exit Reasons (Count)</h2>
        <canvas id="exitChart" height="120"></canvas>
    </div>
    <div class="chart-container">
        <h2>Exit Reasons (P&L)</h2>
        <canvas id="exitPnlChart" height="120"></canvas>
    </div>
</div>

<!-- Recent Trades Table -->
<div class="chart-container">
    <h2>Trade Log (Last 50)</h2>
    <table>
        <thead>
            <tr>
                <th>Symbol</th><th>Direction</th><th>Entry</th><th>Exit</th>
                <th>P&L %</th><th>P&L $</th><th>Days</th><th>Reason</th>
            </tr>
        </thead>
        <tbody id="tradeTable"></tbody>
    </table>
</div>

<div class="footer">
    AlphaDesk Quantitative Trading System — Backtest Report<br>
    Past performance does not guarantee future results. Use Demo environment for testing.
</div>

<script>
const chartDefaults = {{
    color: '#94a3b8',
    borderColor: '#334155',
    font: {{ family: '-apple-system, sans-serif' }}
}};
Chart.defaults.color = '#94a3b8';
Chart.defaults.borderColor = '#334155';

// Equity Curve
new Chart(document.getElementById('equityChart'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(equity_dates)},
        datasets: [{{
            label: 'Portfolio Equity',
            data: {json.dumps(equity_values)},
            borderColor: '#60a5fa',
            backgroundColor: 'rgba(96,165,250,0.1)',
            fill: true,
            pointRadius: 0,
            borderWidth: 2,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ display: true, ticks: {{ maxTicksLimit: 12 }} }},
            y: {{ display: true, ticks: {{ callback: v => '$' + v.toLocaleString() }} }}
        }}
    }}
}});

// Drawdown
new Chart(document.getElementById('drawdownChart'), {{
    type: 'line',
    data: {{
        labels: {json.dumps(equity_dates)},
        datasets: [{{
            label: 'Drawdown %',
            data: {json.dumps(drawdowns)},
            borderColor: '#f87171',
            backgroundColor: 'rgba(248,113,113,0.15)',
            fill: true,
            pointRadius: 0,
            borderWidth: 1.5,
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{
            x: {{ display: true, ticks: {{ maxTicksLimit: 12 }} }},
            y: {{ reverse: true, ticks: {{ callback: v => v.toFixed(1) + '%' }} }}
        }}
    }}
}});

// Monthly P&L
new Chart(document.getElementById('monthlyChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(monthly_labels)},
        datasets: [{{
            label: 'Monthly P&L',
            data: {json.dumps(monthly_values)},
            backgroundColor: {json.dumps(monthly_colors)},
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
        scales: {{ y: {{ ticks: {{ callback: v => '$' + v.toLocaleString() }} }} }}
    }}
}});

// Distribution
const bins = {{}};
const pnls = {json.dumps(pnl_pcts)};
pnls.forEach(p => {{
    const bin = Math.round(p);
    bins[bin] = (bins[bin] || 0) + 1;
}});
const sortedBins = Object.keys(bins).sort((a,b) => a-b);
new Chart(document.getElementById('distChart'), {{
    type: 'bar',
    data: {{
        labels: sortedBins.map(b => b + '%'),
        datasets: [{{
            data: sortedBins.map(b => bins[b]),
            backgroundColor: sortedBins.map(b => b >= 0 ? 'rgba(34,197,94,0.6)' : 'rgba(239,68,68,0.6)'),
        }}]
    }},
    options: {{
        responsive: true,
        plugins: {{ legend: {{ display: false }} }},
    }}
}});

// Exit reasons
new Chart(document.getElementById('exitChart'), {{
    type: 'doughnut',
    data: {{
        labels: {json.dumps(exit_labels)},
        datasets: [{{ data: {json.dumps(exit_counts)}, backgroundColor: ['#60a5fa','#34d399','#f87171','#fbbf24','#a78bfa'] }}]
    }},
    options: {{ responsive: true }}
}});

new Chart(document.getElementById('exitPnlChart'), {{
    type: 'bar',
    data: {{
        labels: {json.dumps(exit_labels)},
        datasets: [{{
            data: {json.dumps(exit_pnls)},
            backgroundColor: {json.dumps(exit_pnls)}.map(v => v >= 0 ? 'rgba(34,197,94,0.7)' : 'rgba(239,68,68,0.7)'),
        }}]
    }},
    options: {{
        responsive: true,
        indexAxis: 'y',
        plugins: {{ legend: {{ display: false }} }},
    }}
}});

// Trade table
const trades = {json.dumps([{
    "s": t.symbol, "d": t.direction,
    "en": t.entry_date.strftime("%Y-%m-%d") if isinstance(t.entry_date, (datetime, pd.Timestamp)) else str(t.entry_date)[:10],
    "ex": t.exit_date.strftime("%Y-%m-%d") if isinstance(t.exit_date, (datetime, pd.Timestamp)) else str(t.exit_date)[:10],
    "pp": round(t.pnl_pct * 100, 2), "p": round(t.pnl, 2),
    "days": t.holding_days, "r": t.exit_reason
} for t in result.trades[-50:]])};

const tbody = document.getElementById('tradeTable');
trades.reverse().forEach(t => {{
    const cls = t.p >= 0 ? 'positive' : 'negative';
    tbody.innerHTML += `<tr>
        <td>${{t.s}}</td><td>${{t.d}}</td><td>${{t.en}}</td><td>${{t.ex}}</td>
        <td class="${{cls}}">${{t.pp > 0 ? '+' : ''}}${{t.pp}}%</td>
        <td class="${{cls}}">${{t.p > 0 ? '+' : ''}}$${{t.p.toLocaleString()}}</td>
        <td>${{t.days}}</td><td>${{t.r}}</td>
    </tr>`;
}});
</script>
</body>
</html>"""

    with open(output_path, "w") as f:
        f.write(html)

    return output_path
