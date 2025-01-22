import os
import json
import logging
import sys
from datetime import datetime
from typing import Tuple, Dict, List, Union
from dotenv import load_dotenv
import torch
import torch.nn as nn
import numpy as np
import requests
from sklearn.metrics import r2_score, mean_absolute_error
from scrapegraphai.graphs import SmartScraperGraph
from fastapi import FastAPI, HTTPException
import matplotlib.pyplot as plt
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

# Initialize Rich console for pretty output
console = Console()

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Load configuration with error handling
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    console.print("[red]Error: config.json not found. Please ensure the configuration file exists.[/red]")
    sys.exit(1)
except json.JSONDecodeError:
    console.print("[red]Error: Invalid JSON in config.json. Please check the file format.[/red]")
    sys.exit(1)

# Configure logging with timestamps
log_filename = f"bundestag_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler(),
    ],
)

# Constants
TOTAL_SEATS = 630
DEFAULT_PARTIES = ["CDU/CSU", "SPD", "Grüne", "FDP", "AfD", "Linke", "Others"]
DEFAULT_POLLS = [30.5, 20.0, 18.5, 8.0, 12.0, 6.0, 5.0]

class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass

class SeatPredictionModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SeatPredictionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def validate_polling_data(parties: List[str], polling_percentages: List[float]) -> None:
    """Validate polling data with detailed error messages."""
    try:
        if not parties or not polling_percentages:
            raise ValidationError("Empty polling data received")
        
        if len(parties) != len(polling_percentages):
            raise ValidationError(f"Mismatched lengths: {len(parties)} parties vs {len(polling_percentages)} percentages")
        
        if not all(isinstance(p, str) for p in parties):
            raise ValidationError("Invalid party names: all names must be strings")
        
        if not all(isinstance(p, (int, float)) for p in polling_percentages):
            raise ValidationError("Invalid polling percentages: all values must be numbers")
        
        if not all(0 <= p <= 100 for p in polling_percentages):
            invalid_values = [p for p in polling_percentages if not 0 <= p <= 100]
            raise ValidationError(f"Polling percentages out of range [0-100]: {invalid_values}")
        
        total = sum(polling_percentages)
        if not 99 <= total <= 101:  # Allow for small rounding errors
            raise ValidationError(f"Total percentage ({total}%) is not close to 100%")
            
        console.print("[green]✓ Polling data validated successfully[/green]")
    except ValidationError as e:
        console.print(f"[red]Validation Error: {str(e)}[/red]")
        raise

def fetch_polling_data() -> Tuple[List[str], List[float]]:
    """Fetch polling data with progress indication and error handling."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching polling data...", total=None)
        
        try:
            if not MISTRAL_API_KEY:
                console.print(Panel(
                    "[yellow]No Mistral API key found. Using default polling data.[/yellow]\n"
                    "To use real-time data, please set MISTRAL_API_KEY in your .env file.",
                    title="Warning"
                ))
                return DEFAULT_PARTIES, DEFAULT_POLLS

            prompt = "Extract the latest polling percentages for German political parties."
            source_url = config.get("polling_url")
            
            if not source_url:
                raise ValueError("Polling URL not found in config")

            progress.update(task, description="Using Mistral AI for data extraction...")
            graph_config = {
                "llm": {
                    "api_key": MISTRAL_API_KEY,
                    "model": "mistral-large",
                },
                "verbose": True,
            }

            scraper = SmartScraperGraph(prompt=prompt, source=source_url, config=graph_config)
            result = scraper.run()

            if not result or 'polls' not in result:
                raise ValueError("Invalid response format from scraper")

            parties = [entry['party'] for entry in result['polls']]
            polling_percentages = [entry['percentage'] for entry in result['polls']]

            validate_polling_data(parties, polling_percentages)
            progress.update(task, description="✓ Polling data fetched successfully")
            
            return parties, polling_percentages

        except requests.exceptions.RequestException as e:
            console.print(f"[red]Network Error: Failed to fetch polling data - {str(e)}[/red]")
            console.print("[yellow]Falling back to default polling data[/yellow]")
            return DEFAULT_PARTIES, DEFAULT_POLLS
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            console.print("[yellow]Falling back to default polling data[/yellow]")
            return DEFAULT_PARTIES, DEFAULT_POLLS

def preprocess_data(polling_percentages: List[float]) -> np.ndarray:
    """Preprocess polling data with error handling."""
    try:
        console.print("Preprocessing polling data...")
        total_percentage = sum(polling_percentages)
        if total_percentage == 0:
            raise ValueError("Total percentage cannot be zero")
            
        normalized_data = [p / total_percentage for p in polling_percentages]
        console.print("[green]✓ Data preprocessing completed[/green]")
        return np.array(normalized_data, dtype=np.float32)
    except Exception as e:
        console.print(f"[red]Error preprocessing data: {str(e)}[/red]")
        raise

def train_model(input_data: np.ndarray) -> SeatPredictionModel:
    """Train the model with progress tracking and error handling."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Training model...", total=100)
        
        try:
            input_size = len(input_data)
            hidden_size = config["model"].get("hidden_size", 64)
            output_size = config["model"].get("output_size", 1)
            learning_rate = config["model"].get("learning_rate", 0.001)
            epochs = config["model"].get("epochs", 1000)

            model = SeatPredictionModel(input_size, hidden_size, output_size)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            targets = torch.tensor([TOTAL_SEATS], dtype=torch.float32).unsqueeze(0)

            model.train()
            for epoch in range(epochs):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                progress.update(task, completed=(epoch + 1) * 100 // epochs)
                
                if (epoch + 1) % 100 == 0:
                    console.print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

            console.print("[green]✓ Model training completed successfully[/green]")
            return model
        except Exception as e:
            console.print(f"[red]Error training model: {str(e)}[/red]")
            raise ModelError(f"Model training failed: {str(e)}")

def predict_seats(
    model: SeatPredictionModel,
    input_data: np.ndarray,
    parties: List[str]
) -> Dict[str, int]:
    """Predict seat distribution with error handling."""
    console.print("Predicting seat distribution...")
    try:
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            total_seats_pred = model(inputs).item()

        seat_allocations = [round(p * total_seats_pred) for p in input_data]

        # Adjust for rounding errors
        seat_difference = TOTAL_SEATS - sum(seat_allocations)
        if seat_difference != 0:
            max_index = seat_allocations.index(max(seat_allocations))
            seat_allocations[max_index] += seat_difference

        seat_distribution = dict(zip(parties, seat_allocations))
        
        # Validate results
        total_allocated = sum(seat_distribution.values())
        if total_allocated != TOTAL_SEATS:
            raise ValueError(f"Invalid seat allocation: total {total_allocated} != {TOTAL_SEATS}")

        console.print("[green]✓ Seat prediction completed successfully[/green]")
        return seat_distribution
    except Exception as e:
        console.print(f"[red]Error predicting seats: {str(e)}[/red]")
        raise

def plot_seat_distribution(seat_distribution: Dict[str, int]) -> None:
    """Plot seat distribution with error handling."""
    try:
        console.print("Generating visualization...")
        parties = list(seat_distribution.keys())
        seats = list(seat_distribution.values())

        plt.figure(figsize=(12, 7))
        bars = plt.bar(parties, seats)
        plt.xlabel("Parties")
        plt.ylabel("Seats")
        plt.title("Bundestagswahl 2025: Predicted Seat Distribution")
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"seat_distribution_{timestamp}.png"
        plt.savefig(filename)
        console.print(f"[green]✓ Plot saved as {filename}[/green]")
        
        plt.show()
    except Exception as e:
        console.print(f"[red]Error generating plot: {str(e)}[/red]")
        raise

# FastAPI setup
app = FastAPI(title="Bundestag Seat Prediction API")

@app.get("/predict")
async def predict_endpoint():
    """API endpoint with error handling."""
    try:
        console.print("\n[bold]Starting prediction process...[/bold]")
        parties, polling_percentages = fetch_polling_data()
        input_data = preprocess_data(polling_percentages)
        model = train_model(input_data)
        seat_distribution = predict_seats(model, input_data, parties)
        return {"success": True, "data": seat_distribution}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main function with comprehensive error handling and progress reporting."""
    console.print(Panel.fit(
        "[bold blue]Bundestag Seat Prediction Tool[/bold blue]\n"
        "This tool predicts seat distribution based on polling data.",
        title="Welcome"
    ))
    
    try:
        # Check environment setup
        if not MISTRAL_API_KEY:
            console.print(Panel(
                "[yellow]Running in demo mode with default data.[/yellow]\n"
                "Set up Mistral API key in .env file for real-time predictions.",
                title="Notice"
            ))

        parties, polling_percentages = fetch_polling_data()
        
        console.print("\n[bold]Current Polling Data:[/bold]")
        for party, percentage in zip(parties, polling_percentages):
            console.print(f"{party}: {percentage}%")

        input_data = preprocess_data(polling_percentages)
        model = train_model(input_data)
        seat_distribution = predict_seats(model, input_data, parties)

        console.print("\n[bold]Predicted Seat Distribution:[/bold]")
        for party, seats in seat_distribution.items():
            console.print(f"{party}: {seats} seats")

        plot_seat_distribution(seat_distribution)
        
        console.print("\n[green]✓ Analysis completed successfully![/green]")
        console.print(f"[blue]Log file saved as: {log_filename}[/blue]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Process interrupted by user[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[red]Critical error: {str(e)}[/red]")
        logging.error(f"Critical error: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()