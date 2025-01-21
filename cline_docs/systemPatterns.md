# System Patterns

## Architecture
The system follows a modular architecture with these key components:

1. Data Collection Layer
   - Uses ScrapeGraphAI for automated polling data extraction
   - OpenAI GPT-3.5 for intelligent data parsing
   - Validation checks for data integrity

2. Machine Learning Layer
   - PyTorch-based neural network model
   - Two-layer architecture (input -> hidden -> output)
   - Adam optimizer with MSE loss function
   - Normalized input processing

3. API Layer
   - FastAPI for RESTful endpoint exposure
   - Single prediction endpoint (/predict)
   - JSON response format

4. Visualization Layer
   - Matplotlib for seat distribution plotting
   - Bar chart representation
   - Interactive display capability

## Key Technical Decisions
1. Neural Network Architecture
   - Simple feedforward network chosen for interpretability
   - ReLU activation for non-linearity
   - Single hidden layer sufficient for this prediction task

2. Error Handling
   - Comprehensive logging system
   - Multiple log handlers (file and stream)
   - Separate error and info logs

3. Data Processing
   - Percentage normalization for consistent scaling
   - Automatic seat total adjustment to maintain 630 total
   - Type validation for inputs

4. Configuration Management
   - Environment variables for sensitive data
   - JSON config file for model parameters
   - dotenv for local development
