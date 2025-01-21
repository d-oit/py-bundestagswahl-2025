# Product Context

## Purpose
This project aims to predict seat distribution in the 2025 German Bundestag election (Bundestagswahl) based on polling data. 

## Problems Solved
1. Automated polling data collection using ScrapeGraphAI
2. Real-time seat prediction using machine learning
3. Visual representation of predicted seat distribution
4. API access to predictions through FastAPI endpoint

## How It Works
1. Fetches latest polling data for German political parties
2. Preprocesses and normalizes the polling percentages
3. Uses a neural network model to predict seat distribution
4. Ensures the total seats sum to the required 630 seats
5. Provides both visual plots and API access to predictions
6. Includes comprehensive logging and error handling
