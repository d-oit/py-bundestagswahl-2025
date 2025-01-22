# Technical Context

## Technologies Used

### Core Dependencies
- Python 3.8+
- PyTorch (ML model)
- FastAPI (API server)
- Plotly (interactive visualizations)
- numpy (numerical operations)
- scikit-learn (data processing)
- python-dotenv (environment management)

### External Services
- Mistral AI API (primary data extraction)
- OpenAI API (fallback data extraction)
- Dawum API (historical data)
- ScrapeGraphAI (web scraping)

### Development Tools
- Git (version control)
- Virtual Environment (Python environment isolation)
- VSCode (recommended IDE)

## Architecture Components

### Data Collection Layer
- Primary: Dawum API client
- Fallback: AI-powered web scraping
  - Mistral AI integration
  - OpenAI fallback system
- Historical data aggregation

### Processing Layer
#### Seat Allocation Methods
1. D'Hondt Method
   - Favors larger parties slightly
   - Default allocation method
   - Implementation in SeatAllocationMethod class

2. Sainte-Laguë Method
   - More proportional allocation
   - Alternative method
   - Configurable via settings

#### Data Processing
- 5% threshold handling
- Percentage normalization
- Historical trend analysis
- Coalition compatibility checking

### Visualization Layer
- Interactive Plotly dashboards
  - Seat distribution comparisons
  - Historical trends
  - Coalition analysis
- Real-time updates
- HTML output generation
- Custom color schemes

### Machine Learning Component
- Neural network model (PyTorch)
- Historical data training
- Trend prediction
- Model persistence

## Development Setup

### Environment Configuration
1. Python virtual environment required
2. `.env` file configuration:
   ```
   MISTRAL_API_KEY=your-key
   OPENAI_API_KEY=your-key
   DAWUM_API_URL=endpoint
   ```

### Required Files
1. `config.json` - Configuration settings
   - Model parameters
   - API endpoints
   - Visualization settings
2. `requirements.txt` - Python dependencies
3. `.env` - Environment variables
4. Log files:
   - `info.log` - General logging
   - `errors.log` - Error tracking
   - `prediction_{timestamp}.log` - Prediction records

## Technical Constraints

### System Requirements
- Python 3.8 or higher
- Minimum 4GB RAM for PyTorch model
- Internet connection for:
  - Real-time polling data
  - Historical data fetching
  - AI API access
- Storage space for:
  - Interactive visualization files
  - Historical data cache
  - Log files

### Data Constraints
- Input polling data:
  - Must sum to approximately 100%
  - Party names must be strings
  - Percentages between 0-100
- Historical data:
  - Minimum 2 years of data points
  - Consistent party naming
- Seat allocation:
  - Total seats fixed at 630
  - 5% threshold rule
  - Valid party identifiers

### Performance Considerations
1. Model Training
   - Occurs on each prediction
   - Configurable epoch count
   - Batch size optimization

2. Data Processing
   - Parallel processing for multiple allocation methods
   - Caching of historical data
   - Efficient memory management for large datasets

3. Visualization Generation
   - Asynchronous plot generation
   - Optimized HTML file size
   - Browser compatibility considerations

4. API Response Times
   - Data scraping latency
   - Model training duration
   - Plot generation overhead
   - Network conditions

### Scaling Considerations
- Historical data storage growth
- Visualization file management
- API request handling
- Memory usage optimization

## Error Handling
1. Data Validation
   - Polling data integrity checks
   - Historical data consistency
   - Configuration validation

2. Failover Systems
   - AI service fallback chain
   - Default data fallback
   - Visualization fallback options

3. Error Reporting
   - Structured logging
   - Error classification
   - Debug information capture
