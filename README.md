# Bundestagswahl 2025 Predictor

A Python-based machine learning system that predicts seat distribution in the 2025 German Bundestag election based on current polling data.

## Features

- 🤖 Automated polling data collection using ScrapeGraphAI
- 🧮 Real-time seat prediction using neural networks
- 📊 Visual representation of predicted seat distribution
- 🚀 FastAPI endpoint for programmatic access
- 📝 Comprehensive logging and error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/py-bundestagswahl-2025.git
cd py-bundestagswahl-2025
```

2. Create and activate a Python virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key
```

## Usage

### CLI with Visualization

Run the application with visualization:
```bash
python main.py
```

This will:
1. Fetch latest polling data
2. Process and normalize the data
3. Generate seat predictions
4. Display a visual representation

### API Server

Start the FastAPI server:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000` with the following endpoint:
- `/predict` - GET endpoint that returns predicted seat distribution

## Technical Details

### Architecture

1. **Data Collection Layer**
   - ScrapeGraphAI for polling data extraction
   - OpenAI GPT-3.5 for data parsing
   - Data validation and integrity checks

2. **Machine Learning Layer**
   - PyTorch neural network model
   - Two-layer architecture
   - Normalized input processing

3. **API Layer**
   - FastAPI RESTful endpoint
   - JSON response format

4. **Visualization Layer**
   - Matplotlib-based plotting
   - Interactive display capability

### System Requirements

- Python 3.x
- Sufficient memory for PyTorch model
- Internet connection
- OpenAI API access

### Dependencies

Core dependencies include:
- PyTorch
- FastAPI
- numpy
- scikit-learn
- matplotlib
- python-dotenv
- ScrapeGraphAI


