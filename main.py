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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

class SeatAllocationMethod:
    """Implements different seat allocation methods for the Bundestag."""

    @staticmethod
    def dhondt_method(percentages: List[float], total_seats: int) -> List[int]:
        """
        Implement D'Hondt method for seat allocation.
        This method tends to favor larger parties slightly.
        """
        seats = [0] * len(percentages)
        divisors = [1] * len(percentages)

        for _ in range(total_seats):
            quotients = [p / d for p, d in zip(percentages, divisors)]
            max_idx = quotients.index(max(quotients))
            seats[max_idx] += 1
            divisors[max_idx] += 1

        return seats

    @staticmethod
    def sainte_lague_method(percentages: List[float], total_seats: int) -> List[int]:
        """
        Implement Sainte-Laguë method for seat allocation.
        This method is considered more proportional than D'Hondt.
        """
        seats = [0] * len(percentages)
        divisors = [1] * len(percentages)

        for _ in range(total_seats):
            quotients = [p / (2 * d - 1) for p, d in zip(percentages, divisors)]
            max_idx = quotients.index(max(quotients))
            seats[max_idx] += 1
            divisors[max_idx] += 1

        return seats

def predict_seats(
    input_percentages: List[float],
    parties: List[str],
    method: str = "dhondt"
) -> Dict[str, int]:
    """
    Predict seat distribution using specified allocation method.

    Args:
        input_percentages: List of party vote percentages
        parties: List of party names
        method: Allocation method ('dhondt' or 'sainte_lague')
    """
    console.print("[bold cyan]Calculating seat distribution...[/bold cyan]")

    try:
        # Apply threshold rule (5% threshold in Germany)
        threshold = 5.0
        valid_indices = [i for i, p in enumerate(input_percentages) if p >= threshold]

        if not valid_indices:
            raise ValueError("No party passed the 5% threshold")

        # Recalculate percentages for parties that passed threshold
        valid_percentages = [input_percentages[i] for i in valid_indices]
        total_valid = sum(valid_percentages)
        normalized_percentages = [p * 100 / total_valid for p in valid_percentages]

        # Select allocation method
        if method.lower() == "sainte_lague":
            seat_allocations = SeatAllocationMethod.sainte_lague_method(normalized_percentages, TOTAL_SEATS)
        else:  # default to D'Hondt
            seat_allocations = SeatAllocationMethod.dhondt_method(normalized_percentages, TOTAL_SEATS)

        # Create final distribution
        valid_parties = [parties[i] for i in valid_indices]
        seat_distribution = dict(zip(valid_parties, seat_allocations))

        # Add zero seats for parties below threshold
        for i, party in enumerate(parties):
            if i not in valid_indices:
                seat_distribution[party] = 0

        console.print(f"[green]✓ Seat prediction successfully computed using {method} method.[/green]")
        return seat_distribution
    except Exception as e:
        console.print(f"[red]Error in seat prediction: {e}[/red]")
        raise

def fetch_historical_data(start_year: int = 2021, end_year: int = 2024) -> Dict[str, Dict[str, float]]:
    """Fetch historical polling data for trend analysis."""
    console.print("[bold cyan]Fetching historical data...[/bold cyan]")

    try:
        # In a real implementation, this would fetch from the Dawum API
        # For now, using example historical data
        historical_data = {
            "CDU/CSU": {
                "2021": 24.1, "2022": 28.2, "2023": 30.5, "2024": 31.0
            },
            "SPD": {
                "2021": 25.7, "2022": 22.1, "2023": 20.0, "2024": 19.5
            },
            "Grüne": {
                "2021": 14.8, "2022": 16.5, "2023": 18.5, "2024": 17.8
            },
            "FDP": {
                "2021": 11.5, "2022": 9.0, "2023": 8.0, "2024": 7.5
            },
            "AfD": {
                "2021": 10.3, "2022": 11.8, "2023": 12.0, "2024": 12.5
            },
            "Linke": {
                "2021": 4.9, "2022": 5.5, "2023": 6.0, "2024": 5.8
            }
        }
        console.print("[green]✓ Historical data fetched successfully[/green]")
        return historical_data
    except Exception as e:
        console.print(f"[red]Error fetching historical data: {str(e)}[/red]")
        return {}

def plot_interactive_analysis(seat_distributions: Dict[str, Dict[str, int]], historical_data: Dict = None) -> None:
    """Create interactive visualization comparing different allocation methods and historical trends."""
    try:
        console.print("Generating interactive visualization...")

        # Create figure with subplots
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Seat Distribution Comparison', 'Historical Trends'),
            vertical_spacing=0.25
        )

        # Colors for parties
        party_colors = {
            'CDU/CSU': '#000000',  # Black
            'SPD': '#E3000F',      # Red
            'Grüne': '#1AA037',    # Green
            'FDP': '#FFED00',      # Yellow
            'AfD': '#0489DB',      # Blue
            'Linke': '#BE3075',    # Purple
            'Others': '#808080'     # Gray
        }

        # Plot seat distributions for different methods
        for method, distribution in seat_distributions.items():
            parties = list(distribution.keys())
            seats = list(distribution.values())

            fig.add_trace(
                go.Bar(
                    name=method.capitalize(),
                    x=parties,
                    y=seats,
                    marker_color=[party_colors.get(party, '#808080') for party in parties],
                    text=seats,
                    textposition='auto',
                    hovertemplate='<b>%{x}</b><br>Seats: %{y}<br>Method: ' + method
                ),
                row=1, col=1
            )

        # Add historical trends if available
        if historical_data:
            for party in parties:
                if party in historical_data:
                    years = list(historical_data[party].keys())
                    values = list(historical_data[party].values())

                    fig.add_trace(
                        go.Scatter(
                            name=party,
                            x=years,
                            y=values,
                            mode='lines+markers',
                            line=dict(color=party_colors.get(party, '#808080'))
                        ),
                        row=2, col=1
                    )

        # Update layout
        fig.update_layout(
            height=1000,
            barmode='group',
            title_text="Bundestag Election Analysis 2025",
            showlegend=True
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Number of Seats", row=1, col=1)
        fig.update_yaxes(title_text="Support Percentage", row=2, col=1)

        # Save interactive plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = f"election_analysis_{timestamp}.html"
        fig.write_html(html_file)
        console.print(f"[green]✓ Interactive visualization saved as {html_file}[/green]")

    except Exception as e:
        console.print(f"[red]Error generating interactive plot: {str(e)}[/red]")
        raise

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

        # Predict seat distribution using both methods
        dhondt_distribution = predict_seats(polling_percentages, parties, method="dhondt")
        sainte_lague_distribution = predict_seats(polling_percentages, parties, method="sainte_lague")

        seat_distributions = {
            "dhondt": dhondt_distribution,
            "sainte_lague": sainte_lague_distribution
        }

        console.print("\n[bold]Predicted Seat Distribution:[/bold]")
        for method, distribution in seat_distributions.items():
            console.print(f"\n{method.capitalize()} Method:")
            for party, seats in distribution.items():
                console.print(f"{party}: {seats} seats")

        # Fetch historical data
        historical_data = fetch_historical_data()

        # Generate interactive visualization
        plot_interactive_analysis(seat_distributions, historical_data)

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
