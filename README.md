# Global Sales Analytics Dashboard

An interactive business intelligence dashboard built with Dash and Plotly for visualizing and analyzing global sales data.

## Features

- 📊 **Interactive Visualizations**: Monthly trends, forecasts, heatmaps, and more
- 📤 **CSV Upload**: Drag-and-drop interface for uploading sales data
- 🔍 **Dynamic Filtering**: Filter by country, product, and date range
- 📈 **Sales Forecasting**: Exponential Moving Average (EMA) based predictions
- ⚠️ **Anomaly Detection**: Automatic detection of unusual sales patterns using Z-score analysis
- 📄 **Report Generation**: Downloadable business summary reports
- 💎 **Premium UI**: Glassmorphic design with smooth animations

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Venkatesh-Karthik/Data-Visualization.git
cd Data-Visualization
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Run the dashboard application:

```bash
python "new (1).py"
```

The dashboard will start and be accessible at:
- Local: http://127.0.0.1:8050
- Network: http://0.0.0.0:8050

### CSV Data Format

Your CSV file must include the following columns:

- **Order_Date**: Date of the order (format: YYYY-MM-DD or any standard date format)
- **Sales**: Sales amount (numeric)
- **Profit**: Profit amount (numeric)
- **Discount**: Discount percentage (numeric)
- **Country**: Country name (text)
- **Product**: Product name (text)

Example CSV structure:
```csv
Order_Date,Sales,Profit,Discount,Country,Product
2023-01-15,1500.50,450.15,5.0,USA,Product A
2023-01-16,2300.75,690.22,3.5,Canada,Product B
```

### Application Features

1. **Upload Data**: Drag and drop or click to upload your CSV file
2. **Apply Filters**: 
   - Select specific countries and products
   - Choose a date range
   - Toggle forecast horizon (6 or 12 months)
3. **View Visualizations**:
   - KPI cards showing total sales, profit, average discount, and anomalies
   - Monthly sales trend with anomaly markers
   - Sales forecast with trend analysis
   - Product-wise sales comparison
   - Country vs Product heatmap
   - Pie chart (by country or product)
   - Grouped bar chart for detailed comparison
4. **Download Report**: Click the download button to get a text summary

### Error Handling

The dashboard includes comprehensive error handling:

- ✅ CSV validation for required columns
- ✅ Encoding detection for international characters
- ✅ Date parsing with invalid date handling
- ✅ Empty data and edge case protection
- ✅ User-friendly error messages displayed in the UI

## Development

### Debug Mode

The application runs in debug mode by default. To change this:

**Development mode** (with hot reload):
```python
app.run(debug=True, host='0.0.0.0', port=8050)
```

**Production mode**:
```python
app.run(debug=False, host='0.0.0.0', port=8050)
```

### Project Structure

```
Data-Visualization/
├── new (1).py              # Main Dash application
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── assets/                # Static assets
    ├── animations.css     # Animation styles
    ├── background.css     # Background effects
    └── liquid_glass.css   # Glassmorphic UI styles
```

### Key Components

- **App Initialization**: Dash app with Bootstrap styling
- **Layout**: Responsive design with glass morphism effects
- **Callbacks**: 
  - Main dashboard callback for data processing and visualization
  - Download callback for report generation
- **Helper Functions**:
  - `parse_csv()`: CSV parsing with encoding detection
  - `validate_csv_schema()`: Schema validation
  - `create_empty_figure()`: Empty state visualization

## Troubleshooting

### Common Issues

**Problem**: "Missing required columns" error
- **Solution**: Ensure your CSV has all required columns: Order_Date, Sales, Country, Product, Profit, Discount

**Problem**: Date parsing errors
- **Solution**: Ensure dates are in a standard format (YYYY-MM-DD recommended)

**Problem**: No data after filtering
- **Solution**: Adjust your filter selections to include more data

**Problem**: Dashboard won't start
- **Solution**: Check that all dependencies are installed with `pip install -r requirements.txt`

### Getting Help

If you encounter issues:
1. Check the error messages displayed in the dashboard
2. Review the console/terminal output for detailed error logs
3. Ensure your CSV file matches the required format
4. Verify all dependencies are correctly installed

## License

This project is open source and available for educational and commercial use.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

Built with:
- [Dash](https://dash.plotly.com/) - Interactive web applications
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Dash Bootstrap Components](https://dash-bootstrap-components.opensource.faculty.ai/) - Bootstrap styling
- [Pandas](https://pandas.pydata.org/) - Data analysis
- [NumPy](https://numpy.org/) & [SciPy](https://scipy.org/) - Scientific computing
