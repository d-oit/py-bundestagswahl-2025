# Product Context

## Purpose
This project provides comprehensive analysis and prediction of seat distribution in the 2025 German Bundestag election (Bundestagswahl), offering multiple allocation methods, interactive visualizations, and historical trend analysis.

## Key Features

### 1. Multiple Seat Allocation Methods
- **D'Hondt Method**: Traditional approach, slightly favoring larger parties
- **Sainte-Laguë Method**: Alternative method for more proportional representation
- **Automatic Threshold Handling**: Implements 5% threshold rule
- **Comparative Analysis**: Side-by-side comparison of different methods

### 2. Interactive Visualizations
- **Real-time Dashboards**: Interactive Plotly-based visualizations
- **Historical Trends**: Track party support over time
- **Comparative Views**: Multiple allocation methods side by side
- **Coalition Analysis**: Visual representation of possible coalitions

### 3. Data Collection and Analysis
- **Multiple Data Sources**:
  - Dawum API integration
  - AI-powered data extraction (Mistral/OpenAI)
  - Historical data aggregation
- **Trend Analysis**:
  - Historical polling trends
  - Support pattern analysis
  - Coalition probability assessment

### 4. Machine Learning Integration
- Neural network prediction model
- Historical data-based forecasting
- Trend analysis and projections
- Confidence metrics

## Problems Solved

1. **Complex Data Processing**
   - Automated polling data collection
   - Multi-source data integration
   - Historical data management
   - Real-time updates

2. **Seat Allocation Challenges**
   - Multiple allocation method support
   - 5% threshold rule implementation
   - Coalition possibility analysis
   - Error margin handling

3. **Visualization Needs**
   - Interactive data exploration
   - Comparative analysis views
   - Historical trend visualization
   - Real-time updates

4. **Analysis Requirements**
   - Coalition feasibility assessment
   - Historical trend analysis
   - Future projection capabilities
   - Confidence metrics

## How It Works

1. **Data Collection Phase**
   - Fetches current polling data
   - Retrieves historical data
   - Validates and normalizes input
   - Integrates multiple data sources

2. **Processing Phase**
   - Applies selected allocation method(s)
   - Implements threshold rules
   - Calculates seat distributions
   - Analyzes coalition possibilities

3. **Analysis Phase**
   - Processes historical trends
   - Generates predictions
   - Calculates confidence metrics
   - Identifies significant patterns

4. **Visualization Phase**
   - Creates interactive dashboards
   - Generates comparative views
   - Displays historical trends
   - Updates in real-time

## User Benefits

1. **Decision Makers**
   - Compare different allocation methods
   - Analyze historical trends
   - Assess coalition possibilities
   - Make data-driven decisions

2. **Analysts**
   - Access comprehensive data
   - Use interactive visualizations
   - Export detailed reports
   - Track changes over time

3. **Researchers**
   - Access historical data
   - Compare methodologies
   - Analyze trends
   - Generate hypotheses

4. **General Public**
   - Understanding election mechanics
   - Exploring possible outcomes
   - Tracking party support
   - Visualizing trends

## Output Examples

### Seat Distribution Analysis
```
CDU/CSU: 192 seats (D'Hondt) vs 190 seats (Sainte-Laguë)
SPD: 126 seats (D'Hondt) vs 127 seats (Sainte-Laguë)
...
```

### Coalition Analysis
```
Possible Majority Coalitions:
1. Grand Coalition: 318 seats
   - Feasibility: High
   - Historical Precedent: Yes
2. Traffic Light: 308 seats
   - Feasibility: Medium
   - Historical Precedent: Yes
```

### Trend Analysis
```
CDU/CSU Support Trend:
2021: 24.1%
2022: 28.2%
2023: 30.5%
2024: 31.0%
Projection 2025: 31.5% ±2.5%
```

## Future Developments

1. **Enhanced Analytics**
   - More allocation methods
   - Advanced coalition analysis
   - Demographic factor integration
   - Regional analysis

2. **Improved Visualizations**
   - More interactive features
   - Custom view options
   - Export capabilities
   - Mobile optimization

3. **Additional Features**
   - API expansion
   - Real-time notifications
   - Custom reports
   - Data export options
