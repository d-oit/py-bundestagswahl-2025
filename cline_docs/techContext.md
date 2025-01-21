# Technical Context

## Technologies Used

### Core Dependencies
- Python 3.x
- PyTorch
- FastAPI
- numpy
- scikit-learn
- matplotlib
- python-dotenv

### External Services
- OpenAI API
- ScrapeGraphAI

### Development Tools
- Git (version control)
- Virtual Environment (Python environment isolation)
- VSCode (recommended IDE)

## Development Setup

### Environment Configuration
1. Python virtual environment recommended
2. `.env` file required with:
   ```
   OPENAI_API_KEY=your-api-key
   ```

### Required Files
1. `config.json` - Configuration settings
2. `requirements.txt` - Python dependencies
3. `.env` - Environment variables
4. `info.log` & `errors.log` - Log files

## Technical Constraints

### System Requirements
- Python 3.x
- Sufficient memory for PyTorch model
- Internet connection for polling data fetch
- OpenAI API access

### Data Constraints
- Input polling data must sum to approximately 100%
- Party names must be strings
- Polling percentages must be between 0-100
- Total seats fixed at 630

### Performance Considerations
- Model training occurs on each prediction
- API endpoint response time depends on:
  - Data scraping speed
  - Model training time
  - Network latency
