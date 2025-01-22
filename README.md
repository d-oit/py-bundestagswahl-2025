# Bundestagswahl 2025 Prediction

A Python project for predicting and analyzing seat distribution in the 2025 German Bundestag election using advanced statistical methods and interactive visualizations.

## Features

- **Multiple Data Sources**:
  - Primary: dawum.de API (direct polling data)
  - Fallback: AI-powered extraction from wahlrecht.de using:
    - Mistral AI
    - OpenAI (fallback)

- **Advanced Seat Allocation Methods**:
  - D'Hondt method (default)
  - Sainte-Laguë method
  - Automatic 5% threshold handling
  - Coalition possibility analysis

- **Interactive Visualizations**:
  - Real-time interactive dashboards using Plotly
  - Historical trend analysis
  - Comparative seat distribution views
  - Coalition scenarios visualization

- **Machine Learning Integration**:
  - Neural network prediction model
  - Historical data analysis
  - Trend forecasting

## Requirements

- Python 3.8+
- PyTorch
- Plotly
- FastAPI
- Additional requirements in `requirements.txt`

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/py-bundestagswahl-2025.git
cd py-bundestagswahl-2025
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
```

Then edit `.env` and add at least one API key:
- `MISTRAL_API_KEY`: Your Mistral AI API key
- `OPENAI_API_KEY`: Your OpenAI API key (fallback)

## Configuration

The `config.json` file allows customization of:
- Model parameters
- Seat allocation methods
- Historical data range
- API endpoints

Example configuration:
```json
{
    "polling_url": "https://www.wahlrecht.de/umfragen/",
    "dawumApiUrl": "https://api.dawum.de/",
    "model": {
        "hidden_size": 128,
        "learning_rate": 0.001,
        "epochs": 1000
    },
    "seat_allocation": {
        "method": "dhondt",
        "alternatives": ["dhondt", "sainte_lague"]
    }
}
```

## Usage

1. Run the prediction tool:
```bash
python main.py
```

2. Start the API server:
```bash
uvicorn main:app --reload
```

## API Endpoints

- `GET /predict`: Get seat distribution predictions
  - Optional query parameter: `include_analysis=true`
- `GET /analysis`: Get detailed analysis including historical trends

## How It Works

1. **Data Collection**:
   - Fetches latest polling data from dawum.de
   - Falls back to AI extraction if needed (Mistral → OpenAI)

2. **Seat Allocation**:
   - Applies 5% threshold rule
   - Uses configured allocation method (D'Hondt/Sainte-Laguë)
   - Calculates seat distribution

3. **Analysis**:
   - Processes historical trends
   - Analyzes coalition possibilities
   - Generates interactive visualizations

4. **Visualization**:
   - Creates interactive Plotly dashboards
   - Shows comparative analysis between methods
   - Displays historical trends

## Output Examples

### Seat Distribution Analysis
```
Method: D'Hondt
CDU/CSU: 192 seats (30.5%)
SPD: 126 seats (20.0%)
...

Method: Sainte-Laguë
CDU/CSU: 190 seats (30.2%)
SPD: 127 seats (20.2%)
...
```

### Coalition Analysis
```
Possible Coalitions:
1. Grand Coalition (CDU/CSU + SPD):
   - 318 seats (Majority: Yes)
   - Historical compatibility: High
2. Traffic Light (SPD + Grüne + FDP):
   - 308 seats (Majority: Yes)
   - Historical compatibility: Medium
...
```

## Troubleshooting

### Common Issues

1. **API Key Errors**:
   - Ensure either MISTRAL_API_KEY or OPENAI_API_KEY is set in .env
   - Check API key validity

2. **Data Fetching Issues**:
   - Check internet connection
   - Verify API endpoints in config.json
   - System falls back to default data if fetching fails

3. **Visualization Errors**:
   - Ensure plotly is properly installed
   - Check write permissions for HTML output
   - Try clearing browser cache for interactive plots

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure your code:
- Follows PEP 8 style guide
- Includes proper documentation
- Has appropriate test coverage
- Updates relevant documentation

## License

MIT License - see LICENSE file for details
