import base64
import io
import pandas as pd
import numpy as np

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# ================= APP INIT =================
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True
)
app.title = "Global Sales Analytics Dashboard"

# ================= LAYOUT =================
app.layout = html.Div([
    # Background effects
    html.Div(className="glow-orb glow-orb-1"),
    html.Div(className="glow-orb glow-orb-2"),
    html.Div(className="glow-orb glow-orb-3"),
    html.Div(id="noise-overlay"),
    html.Div(className="scanlines"),
    html.Div(className="vignette"),
    
    dbc.Container(fluid=True, className="glass-container stagger-container", children=[

        # ---------- HEADER (Floating Glass Navigation) ----------
        html.Div([
            html.H2("🌍 Global Sales Analytics Dashboard", className="glass-title"),
            html.P("Interactive Business Intelligence & Forecasting System",
                   className="glass-subtitle"),
            html.Div(className="glass-separator")
        ], className="glass-nav shimmer-container animate-fade-in-up animate-delay-1"),

        # ---------- UPLOAD (Holographic Glass Surface) ----------
        html.Div([
            dcc.Upload(
                id="upload-data",
                children=html.Div([
                    html.Span("📤 Drag & Drop or Click to Upload CSV", className="glass-text-primary")
                ]),
                style={
                    "width": "100%",
                    "height": "80px",
                    "lineHeight": "80px",
                    "textAlign": "center",
                    "cursor": "pointer"
                },
                multiple=False,
                className="pulse-border breathing"
            ),
            html.Div(id="upload-status", className="mt-3 glass-text-secondary text-center")
        ], className="glass-upload glass-mb animate-fade-in-up animate-delay-2"),

        # ---------- FILTERS (Cockpit Glass Panel) ----------
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("Country", className="glass-text-secondary"),
                    html.Div([
                        dcc.Dropdown(id="country-filter", multi=True)
                    ], className="glass-capsule")
                ], md=3),

                dbc.Col([
                    html.Label("Product", className="glass-text-secondary"),
                    html.Div([
                        dcc.Dropdown(id="product-filter", multi=True)
                    ], className="glass-capsule")
                ], md=3),

                dbc.Col([
                    html.Label("Date Range", className="glass-text-secondary"),
                    html.Div([
                        dcc.DatePickerRange(id="date-filter")
                    ], className="glass-capsule")
                ], md=3),

                dbc.Col([
                    html.Label("Forecast Horizon", className="glass-text-secondary"),
                    html.Div([
                        dcc.RadioItems(
                            id="forecast-horizon",
                            options=[
                                {"label": "6 Months", "value": 6},
                                {"label": "12 Months", "value": 12},
                            ],
                            value=6,
                            inline=True,
                            className="glass-text-primary"
                        )
                    ], className="glass-capsule")
                ], md=3),
            ], className="g-3")
        ], className="glass-filter-panel glass-mb animate-fade-in-up animate-delay-3"),

        # ---------- KPIs (Ultra-Premium Fintech Tiles) ----------
        dbc.Row(id="kpi-cards", className="glass-mb animate-fade-in-up animate-delay-4"),

        # ---------- TREND + FORECAST (Elevated Glass Panels) ----------
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Monthly Sales Trend", className="glass-chart-header glass-text-primary"),
                    dcc.Graph(id="monthly-trend", config={"displayModeBar": False}),
                    html.Div(className="chart-depth-overlay")
                ], className="glass-chart-panel hover-float depth-hover")
            ], md=6),
            dbc.Col([
                html.Div([
                    html.Div("Sales Forecast", className="glass-chart-header glass-text-primary"),
                    dcc.Graph(id="forecast-chart", config={"displayModeBar": False}),
                    html.Div(className="chart-depth-overlay")
                ], className="glass-chart-panel hover-float depth-hover")
            ], md=6),
        ], className="glass-mb animate-fade-in-up animate-delay-5"),

        # ---------- PRODUCT + HEATMAP (Elevated Glass Panels) ----------
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Product-wise Sales", className="glass-chart-header glass-text-primary"),
                    dcc.Graph(id="product-sales", config={"displayModeBar": False}),
                    html.Div(className="chart-depth-overlay")
                ], className="glass-chart-panel hover-float depth-hover")
            ], md=6),
            dbc.Col([
                html.Div([
                    html.Div("Country vs Product Heatmap", className="glass-chart-header glass-text-primary"),
                    dcc.Graph(id="heatmap", config={"displayModeBar": False}),
                    html.Div(className="chart-depth-overlay")
                ], className="glass-chart-panel hover-float depth-hover")
            ], md=6),
        ], className="glass-mb animate-fade-in-up animate-delay-6"),

        # ---------- PIE + GROUPED BAR (Elevated Glass Panels) ----------
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        dbc.RadioItems(
                            id="pie-mode",
                            options=[
                                {"label": "By Country", "value": "Country"},
                                {"label": "By Product", "value": "Product"},
                            ],
                            value="Country",
                            inline=True,
                            className="glass-text-primary"
                        )
                    ], className="glass-chart-header"),
                    dcc.Graph(id="pie-chart", config={"displayModeBar": False}),
                    html.Div(className="chart-depth-overlay")
                ], className="glass-chart-panel hover-float depth-hover")
            ], md=6),

            dbc.Col([
                html.Div([
                    html.Div("Country vs Product Comparison", className="glass-chart-header glass-text-primary"),
                    dcc.Graph(id="grouped-bar", config={"displayModeBar": False}),
                    html.Div(className="chart-depth-overlay")
                ], className="glass-chart-panel hover-float depth-hover")
            ], md=6),
        ], className="glass-mb animate-fade-in-up animate-delay-7"),

        # ---------- REPORT (Glass Terminal) ----------
        html.Div([
            html.Div("📄 Business Summary Report", className="glass-terminal-header"),
            html.Pre(id="report-text", className="terminal-text"),
            html.Button("⬇️ Download Report", id="download-btn", className="glass-button mt-3"),
            dcc.Download(id="download-report")
        ], className="glass-terminal animate-fade-in-up animate-delay-8")
    ])
])

# ================= HELPERS =================
def parse_csv(contents):
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    return pd.read_csv(io.StringIO(decoded.decode("utf-8")))

# ================= CALLBACK =================
@app.callback(
    [
        Output("upload-status", "children"),
        Output("country-filter", "options"),
        Output("product-filter", "options"),
        Output("country-filter", "value"),
        Output("product-filter", "value"),
        Output("date-filter", "start_date"),
        Output("date-filter", "end_date"),
        Output("kpi-cards", "children"),
        Output("monthly-trend", "figure"),
        Output("forecast-chart", "figure"),
        Output("product-sales", "figure"),
        Output("heatmap", "figure"),
        Output("pie-chart", "figure"),
        Output("grouped-bar", "figure"),
        Output("report-text", "children"),
    ],
    [
        Input("upload-data", "contents"),
        Input("country-filter", "value"),
        Input("product-filter", "value"),
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
        Input("forecast-horizon", "value"),
        Input("pie-mode", "value"),
    ]
)
def update_dashboard(contents, countries, products, start, end, horizon, pie_mode):
    if contents is None:
        empty_fig = {}
        return "", [], [], [], [], None, None, [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""

    df = parse_csv(contents)
    df["Order_Date"] = pd.to_datetime(df["Order_Date"])
    df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)

    country_opts = [{"label": c, "value": c} for c in sorted(df["Country"].unique())]
    product_opts = [{"label": p, "value": p} for p in sorted(df["Product"].unique())]

    countries = countries or df["Country"].unique().tolist()
    products = products or df["Product"].unique().tolist()
    start = start or df["Order_Date"].min()
    end = end or df["Order_Date"].max()

    fdf = df[
        (df["Country"].isin(countries)) &
        (df["Product"].isin(products)) &
        (df["Order_Date"] >= start) &
        (df["Order_Date"] <= end)
    ]

    # ---------- KPIs ----------
    kpis = [
        dbc.Col([
            html.Div([
                html.Div(className="kpi-glow"),
                html.H6("Total Sales", className="glass-text-secondary mb-3"),
                html.H4(f"₹{fdf['Sales'].sum():,.0f}", className="kpi-value")
            ], className="glass-kpi hover-float light-sweep")
        ], md=4),

        dbc.Col([
            html.Div([
                html.Div(className="kpi-glow"),
                html.H6("Total Profit", className="glass-text-secondary mb-3"),
                html.H4(f"₹{fdf['Profit'].sum():,.0f}", className="kpi-value")
            ], className="glass-kpi hover-float light-sweep")
        ], md=4),

        dbc.Col([
            html.Div([
                html.Div(className="kpi-glow"),
                html.H6("Avg Discount", className="glass-text-secondary mb-3"),
                html.H4(f"{fdf['Discount'].mean():.2f}%", className="kpi-value")
            ], className="glass-kpi hover-float light-sweep")
        ], md=4),
    ]

    # ---------- MONTHLY TREND ----------
    monthly = fdf.groupby("Month", as_index=False)["Sales"].sum()
    trend_fig = px.line(monthly, x="Month", y="Sales", markers=True)
    trend_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.1)")
    )
    trend_fig.update_traces(line_color="#3b82f6", marker=dict(color="#60a5fa", size=8))

    # ---------- FORECAST ----------
    monthly["EMA"] = monthly["Sales"].ewm(span=3).mean()
    slope = monthly["EMA"].iloc[-1] - monthly["EMA"].iloc[-2]

    future_vals, last = [], monthly["EMA"].iloc[-1]
    for _ in range(horizon):
        last += slope
        future_vals.append(last)

    future_months = pd.date_range(
        pd.to_datetime(monthly["Month"].iloc[-1]) + pd.offsets.MonthBegin(1),
        periods=horizon, freq="MS"
    ).strftime("%Y-%m")

    forecast_fig = go.Figure()
    forecast_fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Sales"],
                                      mode="lines+markers", name="Actual",
                                      line=dict(color="#3b82f6"), marker=dict(color="#60a5fa", size=8)))
    forecast_fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["EMA"],
                                      mode="lines", name="Trend",
                                      line=dict(color="#8b5cf6", dash="dash")))
    forecast_fig.add_trace(go.Scatter(x=future_months, y=future_vals,
                                      mode="lines+markers", name="Forecast",
                                      line=dict(color="#06b6d4"), marker=dict(color="#22d3ee", size=8)))
    forecast_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.1)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(59,130,246,0.3)")
    )

    # ---------- PRODUCT SALES ----------
    prod_fig = px.bar(
        fdf.groupby("Product", as_index=False)["Sales"].sum(),
        x="Product", y="Sales"
    )
    prod_fig.update_traces(marker_color="#3b82f6")
    prod_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.1)")
    )

    # ---------- HEATMAP ----------
    pivot = pd.pivot_table(fdf, values="Sales", index="Country",
                           columns="Product", aggfunc="sum")
    heat_fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Blues")
    heat_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=20, b=20)
    )

    # ---------- PIE ----------
    pie_df = fdf.groupby(pie_mode, as_index=False)["Sales"].sum()
    pie_fig = px.pie(pie_df, values="Sales", names=pie_mode, hole=0.4)
    pie_fig.update_traces(marker=dict(colors=px.colors.sequential.Blues))
    pie_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(59,130,246,0.3)")
    )

    # ---------- GROUPED BAR ----------
    bar_fig = px.bar(
        pivot.reset_index(),
        x="Country", y=pivot.columns,
        barmode="group"
    )
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.8)"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(59,130,246,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(59,130,246,0.3)")
    )

    # ---------- REPORT ----------
    report = f"""
GLOBAL SALES ANALYTICS REPORT
============================

Total Sales   : ₹{fdf['Sales'].sum():,.0f}
Total Profit  : ₹{fdf['Profit'].sum():,.0f}

Top Country   : {fdf.groupby("Country")["Sales"].sum().idxmax()}
Top Product   : {fdf.groupby("Product")["Sales"].sum().idxmax()}

Generated On  : {pd.Timestamp.now().strftime("%d-%m-%Y %H:%M:%S")}
"""

    return (
        "✅ Data Loaded",
        country_opts, product_opts,
        countries, products,
        start, end,
        kpis,
        trend_fig,
        forecast_fig,
        prod_fig,
        heat_fig,
        pie_fig,
        bar_fig,
        report
    )

# ================= DOWNLOAD =================
@app.callback(
    Output("download-report", "data"),
    Input("download-btn", "n_clicks"),
    State("report-text", "children"),
    prevent_initial_call=True
)
def download_report(_, text):
    return dict(content=text, filename="global_sales_report.txt")

# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)
