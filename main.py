import os
import json
import logging
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
import requests
from sklearn.metrics import r2_score, mean_absolute_error
from scrapegraphai.graphs import SmartScraperGraph
from fastapi import FastAPI
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load configuration
with open("config.json") as f:
    config = json.load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("info.log", mode="w"),
        logging.FileHandler("errors.log", mode="w"),
        logging.StreamHandler(),
    ],
)

# Constants
TOTAL_SEATS = 630

# Define the neural network model
class SeatPredictionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SeatPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# Function to validate polling data
def validate_polling_data(parties, polling_percentages):
    assert len(parties) == len(polling_percentages), "Mismatched lengths"
    assert all(isinstance(p, str) for p in parties), "Invalid party names"
    assert all(0 <= p <= 100 for p in polling_percentages), "Polling percentages out of range"
    logging.info("Polling data validated successfully.")

# Function to fetch polling data using ScrapeGraphAI
def fetch_polling_data():
    try:
        graph_config = {
            "llm": {
                "api_key": OPENAI_API_KEY,
                "model": "openai/gpt-4o-mini",
            },
            "verbose": True,
        }

        prompt = "Extract the latest polling percentages for German political parties."
        source_url = config["polling_url"]

        scraper = SmartScraperGraph(prompt=prompt, source=source_url, config=graph_config)
        result = scraper.run()

        parties = [entry['party'] for entry in result['polls']]
        polling_percentages = [entry['percentage'] for entry in result['polls']]

        validate_polling_data(parties, polling_percentages)

        logging.info("Polling data fetched successfully.")
        return parties, polling_percentages
    except Exception as e:
        logging.error(f"Error fetching polling data: {e}")
        raise

# Function to preprocess data
def preprocess_data(polling_percentages):
    try:
        total_percentage = sum(polling_percentages)
        normalized_data = [p / total_percentage for p in polling_percentages]
        return np.array(normalized_data, dtype=np.float32)
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        raise

# Function to train the model
def train_model(input_data):
    try:
        input_size = len(input_data)
        hidden_size = config["model"]["hidden_size"]
        output_size = config["model"]["output_size"]

        model = SeatPredictionModel(input_size, hidden_size, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config["model"]["learning_rate"])

        inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
        targets = torch.tensor([TOTAL_SEATS], dtype=torch.float32).unsqueeze(0)

        model.train()
        for epoch in range(config["model"]["epochs"]):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 100 == 0:
                logging.info(f'Epoch [{epoch + 1}/{config["model"]["epochs"]}], Loss: {loss.item():.4f}')

        logging.info("Model training completed.")
        return model
    except Exception as e:
        logging.error(f"Error training model: {e}")
        raise

# Function to predict seat allocation
def predict_seats(model, input_data, parties):
    try:
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            total_seats_pred = model(inputs).item()

        seat_allocations = [round(p * total_seats_pred) for p in input_data]

        seat_difference = TOTAL_SEATS - sum(seat_allocations)
        if seat_difference != 0:
            max_index = seat_allocations.index(max(seat_allocations))
            seat_allocations[max_index] += seat_difference

        seat_distribution = dict(zip(parties, seat_allocations))
        return seat_distribution
    except Exception as e:
        logging.error(f"Error predicting seats: {e}")
        raise

# Function to plot seat distribution
def plot_seat_distribution(seat_distribution):
    parties = list(seat_distribution.keys())
    seats = list(seat_distribution.values())

    plt.figure(figsize=(10, 6))
    plt.bar(parties, seats, color='skyblue')
    plt.xlabel("Parties")
    plt.ylabel("Seats")
    plt.title("Bundestagswahl 2025: Predicted Seat Distribution")
    plt.show()

# FastAPI setup
app = FastAPI()

@app.get("/predict")
def predict_endpoint():
    parties, polling_percentages = fetch_polling_data()
    input_data = preprocess_data(polling_percentages)
    model = train_model(input_data)
    seat_distribution = predict_seats(model, input_data, parties)
    return seat_distribution

# Main function
def main():
    try:
        parties, polling_percentages = fetch_polling_data()
        input_data = preprocess_data(polling_percentages)
        model = train_model(input_data)
        seat_distribution = predict_seats(model, input_data, parties)

        logging.info("Predicted seat distribution:")
        for party, seats in seat_distribution.items():
            logging.info(f"{party}: {seats} seats")

        plot_seat_distribution(seat_distribution)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
