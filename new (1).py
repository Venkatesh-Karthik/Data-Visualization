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
    external_stylesheets=[dbc.themes.FLATLY]
)
app.title = "Global Sales Analytics Dashboard"

# ================= LAYOUT =================
app.layout = dbc.Container(fluid=True, children=[

    # ---------- HEADER ----------
    dbc.Row([
        dbc.Col([
            html.H2("🌍 Global Sales Analytics Dashboard", className="fw-bold"),
            html.P("Interactive Business Intelligence & Forecasting System",
                   className="text-muted")
        ])
    ], className="my-3"),

    # ---------- UPLOAD ----------
    dbc.Card([
        dcc.Upload(
            id="upload-data",
            children=html.Div([
                "📤 Drag & Drop or Click to Upload CSV"
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "2px",
                "borderStyle": "dashed",
                "borderRadius": "10px",
                "textAlign": "center",
            },
            multiple=False
        ),
        html.Div(id="upload-status", className="mt-2 text-success")
    ], body=True, className="mb-4"),

    # ---------- FILTERS ----------
    dbc.Card([
        dbc.Row([
            dbc.Col([
                html.Label("Country"),
                dcc.Dropdown(id="country-filter", multi=True)
            ], md=3),

            dbc.Col([
                html.Label("Product"),
                dcc.Dropdown(id="product-filter", multi=True)
            ], md=3),

            dbc.Col([
                html.Label("Date Range"),
                dcc.DatePickerRange(id="date-filter")
            ], md=3),

            dbc.Col([
                html.Label("Forecast Horizon"),
                dcc.RadioItems(
                    id="forecast-horizon",
                    options=[
                        {"label": "6 Months", "value": 6},
                        {"label": "12 Months", "value": 12},
                    ],
                    value=6,
                    inline=True
                )
            ], md=3),
        ], className="g-3")
    ], body=True, className="mb-4"),

    # ---------- KPIs ----------
    dbc.Row(id="kpi-cards", className="mb-4"),

    # ---------- TREND + FORECAST ----------
    dbc.Row([
        dbc.Col(dcc.Graph(id="monthly-trend"), md=6),
        dbc.Col(dcc.Graph(id="forecast-chart"), md=6),
    ], className="mb-4"),

    # ---------- PRODUCT + HEATMAP ----------
    dbc.Row([
        dbc.Col(dcc.Graph(id="product-sales"), md=6),
        dbc.Col(dcc.Graph(id="heatmap"), md=6),
    ], className="mb-4"),

    # ---------- PIE + GROUPED BAR ----------
    dbc.Row([
        dbc.Col([
            dbc.RadioItems(
                id="pie-mode",
                options=[
                    {"label": "By Country", "value": "Country"},
                    {"label": "By Product", "value": "Product"},
                ],
                value="Country",
                inline=True,
                className="mb-2"
            ),
            dcc.Graph(id="pie-chart")
        ], md=6),

        dbc.Col(dcc.Graph(id="grouped-bar"), md=6),
    ], className="mb-4"),

    # ---------- REPORT ----------
    dbc.Card([
        dbc.CardHeader("📄 Business Summary Report"),
        dbc.CardBody([
            html.Pre(id="report-text"),
            dbc.Button("⬇️ Download Report", id="download-btn", color="primary"),
            dcc.Download(id="download-report")
        ])
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
        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Sales"),
            html.H4(f"₹{fdf['Sales'].sum():,.0f}", className="text-primary fw-bold")
        ])), md=4),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Total Profit"),
            html.H4(f"₹{fdf['Profit'].sum():,.0f}", className="text-success fw-bold")
        ])), md=4),

        dbc.Col(dbc.Card(dbc.CardBody([
            html.H6("Avg Discount"),
            html.H4(f"{fdf['Discount'].mean():.2f}%", className="text-warning fw-bold")
        ])), md=4),
    ]

    # ---------- MONTHLY TREND ----------
    monthly = fdf.groupby("Month", as_index=False)["Sales"].sum()
    trend_fig = px.line(monthly, x="Month", y="Sales", markers=True,
                        title="Monthly Sales Trend")

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
                                      mode="lines+markers", name="Actual"))
    forecast_fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["EMA"],
                                      mode="lines", name="Trend"))
    forecast_fig.add_trace(go.Scatter(x=future_months, y=future_vals,
                                      mode="lines+markers", name="Forecast"))
    forecast_fig.update_layout(title="Sales Forecast")

    # ---------- PRODUCT SALES ----------
    prod_fig = px.bar(
        fdf.groupby("Product", as_index=False)["Sales"].sum(),
        x="Product", y="Sales",
        title="Product-wise Sales"
    )

    # ---------- HEATMAP ----------
    pivot = pd.pivot_table(fdf, values="Sales", index="Country",
                           columns="Product", aggfunc="sum")
    heat_fig = px.imshow(pivot, text_auto=".0f",
                          title="Country vs Product Heatmap")

    # ---------- PIE ----------
    pie_df = fdf.groupby(pie_mode, as_index=False)["Sales"].sum()
    pie_fig = px.pie(pie_df, values="Sales", names=pie_mode,
                     hole=0.4, title="Sales Distribution")

    # ---------- GROUPED BAR ----------
    bar_fig = px.bar(
        pivot.reset_index(),
        x="Country", y=pivot.columns,
        barmode="group",
        title="Country vs Product Comparison"
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
