import base64
import io
import pandas as pd
import numpy as np
from scipy import stats
import traceback

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

# ================= CONSTANTS =================
# Required CSV columns for validation
REQUIRED_COLUMNS = ["Order_Date", "Sales", "Country", "Product", "Profit", "Discount"]

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
                    "cursor": "pointer",
                    "pointerEvents": "auto"
                },
                multiple=False,
                className="pulse-border breathing"
            ),
            html.Div(id="upload-status", className="mt-3 glass-text-secondary text-center"),
            # Error message panel for user feedback
            html.Div(id="error-panel", className="mt-3 text-center", style={"color": "#ef4444", "fontSize": "14px"}),
            dcc.Loading(id="upload-loading", type="circle", children=html.Div(id="loading-output"))
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
                        dcc.DatePickerRange(
                            id="date-filter",
                            display_format="DD/MM/YYYY",
                            style={"width": "100%"}
                        )
                    ], className="glass-capsule glass-date-picker")
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
                            className="glass-text-primary glass-radio-segmented",
                            labelStyle={"padding": "12px 20px", "cursor": "pointer", "borderRadius": "12px"}
                        )
                    ], className="glass-capsule")
                ], md=3),
            ], className="g-3")
        ], className="glass-filter-panel glass-mb animate-fade-in-up animate-delay-3"),

        # ---------- KPIs (Ultra-Premium Fintech Tiles) ----------
        dbc.Row(id="kpi-cards", className="glass-mb animate-fade-in-up animate-delay-4"),

        # ---------- TREND + FORECAST (Full Width Stacked Glass Panels) ----------
        html.Div([
            html.Div([
                html.Div("Monthly Sales Trend", className="glass-chart-header glass-text-primary"),
                html.Div(className="glass-separator"),
                dcc.Graph(id="monthly-trend", config={"displayModeBar": False}),
                html.Div(className="chart-depth-overlay")
            ], className="glass-chart-panel hover-float depth-hover glass-mb")
        ], className="animate-fade-in-up animate-delay-5"),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Span("Sales Forecast", className="glass-text-primary"),
                    html.Div(id="anomaly-toggle-container", style={"display": "inline-block", "marginLeft": "20px"}, children=[
                        dcc.Checklist(
                            id="show-anomalies",
                            options=[{"label": " Show Anomalies", "value": "show"}],
                            value=["show"],
                            className="glass-text-secondary",
                            style={"display": "inline-block"}
                        )
                    ])
                ], className="glass-chart-header"),
                html.Div(className="glass-separator"),
                dcc.Graph(id="forecast-chart", config={"displayModeBar": False}),
                html.Div(className="chart-depth-overlay")
            ], className="glass-chart-panel hover-float depth-hover glass-mb")
        ], className="animate-fade-in-up animate-delay-6"),

        # ---------- PRODUCT + HEATMAP (Full Width Stacked Glass Panels) ----------
        html.Div([
            html.Div([
                html.Div("Product-wise Sales", className="glass-chart-header glass-text-primary"),
                html.Div(className="glass-separator"),
                dcc.Graph(id="product-sales", config={"displayModeBar": False}),
                html.Div(className="chart-depth-overlay")
            ], className="glass-chart-panel hover-float depth-hover glass-mb")
        ], className="animate-fade-in-up animate-delay-7"),
        
        html.Div([
            html.Div([
                html.Div("Country vs Product Heatmap", className="glass-chart-header glass-text-primary"),
                html.Div(className="glass-separator"),
                dcc.Graph(id="heatmap", config={"displayModeBar": False}),
                html.Div(className="chart-depth-overlay")
            ], className="glass-chart-panel hover-float depth-hover glass-mb")
        ], className="animate-fade-in-up animate-delay-8"),

        # ---------- PIE + GROUPED BAR (Full Width Stacked Glass Panels) ----------
        html.Div([
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
                html.Div(className="glass-separator"),
                dcc.Graph(id="pie-chart", config={"displayModeBar": False}),
                html.Div(className="chart-depth-overlay")
            ], className="glass-chart-panel hover-float depth-hover glass-mb")
        ], className="animate-fade-in-up animate-delay-9"),

        html.Div([
            html.Div([
                html.Div("Country vs Product Comparison", className="glass-chart-header glass-text-primary"),
                html.Div(className="glass-separator"),
                dcc.Graph(id="grouped-bar", config={"displayModeBar": False}),
                html.Div(className="chart-depth-overlay")
            ], className="glass-chart-panel hover-float depth-hover glass-mb")
        ], className="animate-fade-in-up animate-delay-10"),

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
    """Parse CSV file from base64 encoded contents.
    
    Args:
        contents: Base64 encoded CSV file contents
        
    Returns:
        tuple: (DataFrame, error_message) where error_message is None on success
    """
    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        
        # Try different encodings for robustness
        for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
            try:
                df = pd.read_csv(io.StringIO(decoded.decode(encoding)))
                return df, None
            except UnicodeDecodeError:
                continue
        
        return None, "⚠️ Unable to decode file. Please ensure it's a valid CSV with UTF-8 encoding."
        
    except Exception as e:
        return None, f"⚠️ Error reading CSV: {str(e)}"


def validate_csv_schema(df):
    """Validate that CSV has all required columns.
    
    Args:
        df: Pandas DataFrame to validate
        
    Returns:
        tuple: (is_valid, error_message) where error_message is None if valid
    """
    if df is None or df.empty:
        return False, "⚠️ CSV file is empty. Please upload a file with data."
    
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        return False, f"⚠️ Missing required columns: {', '.join(missing_cols)}. Required: {', '.join(REQUIRED_COLUMNS)}"
    
    return True, None


def create_empty_figure(message="No data available"):
    """Create an empty figure with a message.
    
    Args:
        message: Message to display in the empty figure
        
    Returns:
        Plotly figure object
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16, color="rgba(255,255,255,0.5)")
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig

# ================= CALLBACK =================
@app.callback(
    [
        Output("upload-status", "children"),
        Output("error-panel", "children"),
        Output("loading-output", "children"),
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
        Input("show-anomalies", "value"),
    ],
    prevent_initial_call=True
)
def update_dashboard(contents, countries, products, start, end, horizon, pie_mode, show_anomalies):
    """Main dashboard callback with comprehensive error handling.
    
    Handles CSV upload, filtering, and visualization updates.
    All errors are caught and displayed to the user with clear messages.
    """
    # Return empty state if no file uploaded
    if contents is None:
        empty_fig = create_empty_figure("Upload a CSV file to get started")
        return "", "", "", [], [], [], [], None, None, [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""
    
    try:
        # Parse and validate CSV file
        df, parse_error = parse_csv(contents)
        if parse_error:
            empty_fig = create_empty_figure(parse_error)
            return "", parse_error, "", [], [], [], [], None, None, [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""
        
        # Validate schema
        is_valid, schema_error = validate_csv_schema(df)
        if not is_valid:
            empty_fig = create_empty_figure(schema_error)
            return "", schema_error, "", [], [], [], [], None, None, [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""
        
        # Parse dates with error handling
        try:
            df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors='coerce')
            # Remove rows with invalid dates
            invalid_dates = df["Order_Date"].isna().sum()
            if invalid_dates > 0:
                df = df.dropna(subset=["Order_Date"])
            df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
        except Exception as e:
            error_msg = f"⚠️ Error parsing dates: {str(e)}"
            empty_fig = create_empty_figure(error_msg)
            return "", error_msg, "", [], [], [], [], None, None, [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""
        
        # Verify we have data after cleaning
        if df.empty:
            error_msg = "⚠️ No valid data found after processing. Please check your CSV file."
            empty_fig = create_empty_figure(error_msg)
            return "", error_msg, "", [], [], [], [], None, None, [], empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, ""
        
        # Generate filter options
        country_opts = [{"label": c, "value": c} for c in sorted(df["Country"].unique())]
        product_opts = [{"label": p, "value": p} for p in sorted(df["Product"].unique())]
        
        # Handle None values for filters with defaults
        if countries is None or not countries:
            countries = df["Country"].unique().tolist()
        if products is None or not products:
            products = df["Product"].unique().tolist()
        
        # Clamp date ranges to valid values
        min_date = df["Order_Date"].min()
        max_date = df["Order_Date"].max()
        
        if start is None or pd.to_datetime(start) < min_date:
            start = min_date
        if end is None or pd.to_datetime(end) > max_date:
            end = max_date
        
        # Ensure start <= end
        if pd.to_datetime(start) > pd.to_datetime(end):
            start, end = min_date, max_date
        
        # Apply filters
        fdf = df[
            (df["Country"].isin(countries)) &
            (df["Product"].isin(products)) &
            (df["Order_Date"] >= start) &
            (df["Order_Date"] <= end)
        ]
        
        # Check if filtered data is empty
        if fdf.empty:
            error_msg = "⚠️ No data matches the selected filters. Please adjust your filters."
            empty_fig = create_empty_figure(error_msg)
            return (
                "✅ Data Loaded Successfully",
                error_msg,
                "",
                country_opts, product_opts,
                countries, products,
                start, end,
                [],
                empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                ""
            )

        # ---------- KPIs ----------
        # Detect anomalies in monthly sales using Z-score method
        monthly = fdf.groupby("Month", as_index=False)["Sales"].sum()
        
        # Handle edge case: too few data points for Z-score
        anomaly_count = 0
        anomalies = pd.DataFrame()
        if len(monthly) >= 3:
            try:
                z_scores = np.abs(stats.zscore(monthly["Sales"]))
                anomaly_threshold = 2.0  # Standard threshold for Z-score
                anomalies = monthly[z_scores > anomaly_threshold]
                anomaly_count = len(anomalies)
            except Exception:
                # If Z-score calculation fails, continue without anomalies
                pass
        
        kpis = [
            dbc.Col([
                html.Div([
                    html.Div(className="kpi-glow"),
                    html.H6("Total Sales", className="glass-text-secondary mb-3"),
                    html.H4(f"₹{fdf['Sales'].sum():,.0f}", className="kpi-value")
                ], className="glass-kpi hover-float light-sweep")
            ], md=3),

            dbc.Col([
                html.Div([
                    html.Div(className="kpi-glow"),
                    html.H6("Total Profit", className="glass-text-secondary mb-3"),
                    html.H4(f"₹{fdf['Profit'].sum():,.0f}", className="kpi-value")
                ], className="glass-kpi hover-float light-sweep")
            ], md=3),

            dbc.Col([
                html.Div([
                    html.Div(className="kpi-glow"),
                    html.H6("Avg Discount", className="glass-text-secondary mb-3"),
                    html.H4(f"{fdf['Discount'].mean():.2f}%", className="kpi-value")
                ], className="glass-kpi hover-float light-sweep")
            ], md=3),
            
            dbc.Col([
                html.Div([
                    html.Div(className="kpi-glow"),
                    html.H6("Anomalies", className="glass-text-secondary mb-3"),
                    html.H4(f"{anomaly_count}", className="kpi-value", style={"color": "#ef4444" if anomaly_count > 0 else "#10b981"})
                ], className="glass-kpi hover-float light-sweep")
            ], md=3),
        ]

        # ---------- MONTHLY TREND ----------
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
        
        # Add anomaly markers if toggle is on and anomalies exist
        if show_anomalies and "show" in show_anomalies and len(anomalies) > 0:
            trend_fig.add_trace(go.Scatter(
                x=anomalies["Month"],
                y=anomalies["Sales"],
                mode="markers+text",
                marker=dict(
                    color="#ef4444",
                    size=14,
                    symbol="circle",
                    line=dict(color="#fca5a5", width=2)
                ),
                text=["⚠️ Anomaly"] * len(anomalies),
                textposition="top center",
                textfont=dict(color="#ef4444", size=10),
                name="Anomalies",
                hovertemplate="<b>Anomaly Detected</b><br>Month: %{x}<br>Sales: ₹%{y:,.0f}<extra></extra>"
            ))

        # ---------- FORECAST ----------
        # Ensure we have enough data for forecasting
        if len(monthly) >= 2:
            monthly["EMA"] = monthly["Sales"].ewm(span=min(3, len(monthly))).mean()
            slope = monthly["EMA"].iloc[-1] - monthly["EMA"].iloc[-2]

            # Validate horizon input
            if horizon is None or not isinstance(horizon, (int, float)) or horizon <= 0:
                horizon = 6

            future_vals, last = [], monthly["EMA"].iloc[-1]
            for _ in range(int(horizon)):
                last += slope
                future_vals.append(last)

            future_months = pd.date_range(
                pd.to_datetime(monthly["Month"].iloc[-1]) + pd.offsets.MonthBegin(1),
                periods=int(horizon), freq="MS"
            ).strftime("%Y-%m")
        else:
            # Not enough data for forecasting
            monthly["EMA"] = monthly["Sales"]
            future_vals = []
            future_months = []

        forecast_fig = go.Figure()
        forecast_fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["Sales"],
                                          mode="lines+markers", name="Actual",
                                          line=dict(color="#3b82f6"), marker=dict(color="#60a5fa", size=8)))
        if len(monthly) >= 2:
            forecast_fig.add_trace(go.Scatter(x=monthly["Month"], y=monthly["EMA"],
                                              mode="lines", name="Trend",
                                              line=dict(color="#8b5cf6", dash="dash")))
        if future_vals:
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
        prod_data = fdf.groupby("Product", as_index=False)["Sales"].sum()
        prod_fig = px.bar(prod_data, x="Product", y="Sales")
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
                               columns="Product", aggfunc="sum", fill_value=0)
        heat_fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Blues")
        heat_fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="rgba(255,255,255,0.8)"),
            margin=dict(l=20, r=20, t=20, b=20)
        )

        # ---------- PIE ----------
        # Validate pie_mode input
        if pie_mode not in ["Country", "Product"]:
            pie_mode = "Country"
        
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
            "✅ Data Loaded Successfully",
            "",  # No error
            "",
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
    
    except Exception as e:
        # Catch any unexpected errors and display to user
        error_msg = f"⚠️ Unexpected error: {str(e)}"
        print(f"Dashboard error: {traceback.format_exc()}")  # Log for debugging
        empty_fig = create_empty_figure(error_msg)
        return (
            "",
            error_msg,
            "",
            [], [],
            [], [],
            None, None,
            [],
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            ""
        )

# ================= DOWNLOAD =================
@app.callback(
    Output("download-report", "data"),
    Input("download-btn", "n_clicks"),
    State("report-text", "children"),
    prevent_initial_call=True
)
def download_report(n_clicks, text):
    """Download report callback with error handling.
    
    Args:
        n_clicks: Number of times button was clicked
        text: Report text content
        
    Returns:
        Download data dictionary or None on error
    """
    try:
        # Validate input
        if not text or text.strip() == "":
            return None
        
        return dict(content=text, filename="global_sales_report.txt")
    except Exception as e:
        print(f"Download error: {str(e)}")
        return None

# ================= RUN =================
if __name__ == "__main__":
    # Run the Dash app
    # For development/debugging, set debug=True to enable:
    # - Hot reloading on code changes
    # - Detailed error messages in the browser
    # - Dev tools panel
    # For production, set debug=False
    app.run(debug=True, host='0.0.0.0', port=8050)
    
    # Alternative configurations:
    # app.run(debug=False)  # Production mode
    # app.run(debug=True, dev_tools_hot_reload=False)  # Debug without hot reload
    # app.run(debug=True, host='127.0.0.1', port=8080)  # Custom host/port
