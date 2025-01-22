import os
import json
import logging
import sys
from datetime import datetime
from typing import Tuple, Dict, List, Union
from dataclasses import dataclass
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
from rich.table import Table
from itertools import combinations

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Check if at least one API key is available
if not any([MISTRAL_API_KEY, OPENAI_API_KEY]):
    console.print("[red]Error: At least one API key (Mistral or OpenAI) is required[/red]")
    sys.exit(1)

# Load configuration
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    console.print("[red]Error: config.json not found[/red]")
    sys.exit(1)

# Configure logging
log_filename = f"bundestag_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
)

# Constants
TOTAL_SEATS = 630
DEFAULT_PARTIES = ["CDU/CSU", "SPD", "Grüne", "FDP", "AfD", "Linke", "SSW", "Freie Wähler", "Others"]
DEFAULT_POLLS = [30.5, 20.0, 18.5, 8.0, 12.0, 6.0, 0.5, 2.0, 2.5]

# Coalition compatibility matrix
COALITION_COMPATIBILITY = {
    "CDU/CSU": ["SPD", "FDP", "Grüne"],
    "SPD": ["CDU/CSU", "Grüne", "FDP", "Linke"],
    "Grüne": ["SPD", "CDU/CSU", "FDP", "Linke"],
    "FDP": ["CDU/CSU", "SPD", "Grüne"],
    "Linke": ["SPD", "Grüne"],
    "AfD": [],  # Currently no party forms coalitions with AfD
    "SSW": ["SPD", "Grüne", "FDP"],  # Danish minority party
    "Freie Wähler": ["CDU/CSU"],
    "Others": []
}

# Named coalitions
NAMED_COALITIONS = {
    frozenset(["CDU/CSU", "FDP"]): "Conservative-Liberal Alliance",
    frozenset(["SPD", "Grüne"]): "Red-Green Alliance",
    frozenset(["CDU/CSU", "SPD"]): "Grand Coalition",
    frozenset(["SPD", "Grüne", "FDP"]): "Traffic Light Coalition",
    frozenset(["CDU/CSU", "Grüne", "FDP"]): "Jamaica Coalition",
    frozenset(["SPD", "Grüne", "Linke"]): "Left Alliance",
    frozenset(["CDU/CSU", "Freie Wähler"]): "Conservative Alliance",
    frozenset(["SPD", "Grüne", "SSW"]): "Northern Progressive Alliance"
}

@dataclass
class EconomicIndicators:
    unemployment_rate: float
    gdp_growth: float
    inflation_rate: float

class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass

class ModelError(Exception):
    """Custom exception for model-related errors."""
    pass

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

async def fetch_polling_data() -> Tuple[List[str], List[float]]:
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
            source_url = config.get("pollingUrl")

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

        except Exception as e:
            console.print(f"[red]Error fetching polling data: {str(e)}[/red]")
            return DEFAULT_PARTIES, DEFAULT_POLLS

async def fetch_economic_data() -> EconomicIndicators:
    """Fetch current economic indicators."""
    try:
        # In a real implementation, this would fetch from an economic data API
        # Using example data for demonstration
        return EconomicIndicators(
            unemployment_rate=5.2,  # Example German unemployment rate
            gdp_growth=1.8,        # Example GDP growth rate
            inflation_rate=2.5      # Example inflation rate
        )
    except Exception as e:
        console.print(f"[red]Error fetching economic data: {str(e)}[/red]")
        return EconomicIndicators(5.0, 1.5, 2.0)  # Default values

def preprocess_data(polling_percentages: List[float]) -> np.ndarray:
    """Preprocess polling data with error handling."""
    try:
        console.print("Preprocessing polling data...")

        # Normalize percentages
        total_percentage = sum(polling_percentages)
        if total_percentage == 0:
            raise ValueError("Total percentage cannot be zero")

        normalized_data = np.array([p / total_percentage for p in polling_percentages], dtype=np.float32)

        # Add economic features if available
        if hasattr(preprocess_data, 'economic_data'):
            economic_features = np.array([
                preprocess_data.economic_data.unemployment_rate / 10.0,  # Scale to 0-1 range
                preprocess_data.economic_data.gdp_growth / 5.0,         # Scale to 0-1 range
                preprocess_data.economic_data.inflation_rate / 10.0     # Scale to 0-1 range
            ], dtype=np.float32)
            normalized_data = np.concatenate([normalized_data, economic_features])

        console.print("[green]✓ Data preprocessing completed[/green]")
        return normalized_data

    except Exception as e:
        console.print(f"[red]Error preprocessing data: {str(e)}[/red]")
        raise

class NeuroSymbolicModel(nn.Module):
    """Enhanced model with neuro-symbolic reasoning capabilities."""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(NeuroSymbolicModel, self).__init__()
        # Neural component
        self.neural_net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        # Symbolic rules
        self.symbolic_rules = [
            self._five_percent_threshold,
            self._total_seats_constraint,
            self._economic_impact_rule,
            self._special_party_rules
        ]

    def _five_percent_threshold(self, predictions: torch.Tensor, parties: List[str]) -> torch.Tensor:
        """Symbolic rule: Parties under 5% get no seats unless they win direct mandates or are exempt."""
        mask = torch.ones_like(predictions)
        for i, party in enumerate(parties):
            if party not in ["SSW"]:  # SSW is exempt from 5% threshold
                mask[i] = 1.0 if predictions[i] >= 5.0 else 0.0
        return predictions * mask

    def _total_seats_constraint(self, predictions: torch.Tensor) -> torch.Tensor:
        """Symbolic rule: Ensure total seats sum to TOTAL_SEATS."""
        return predictions * (TOTAL_SEATS / predictions.sum())

    def _economic_impact_rule(self, predictions: torch.Tensor, economic_data: EconomicIndicators) -> torch.Tensor:
        """Symbolic rule: Adjust predictions based on economic indicators."""
        # Create party influence factors based on economic conditions
        factor = torch.ones_like(predictions)

        # Economic status affects different parties differently
        if economic_data.unemployment_rate > 5.0 or economic_data.inflation_rate > 2.0:
            # Poor economic conditions typically favor opposition and populist parties
            factor[predictions.argmax()] *= 0.9  # Reduce leading party's share
            factor[predictions.argmin()] *= 1.1  # Boost smallest party

        if economic_data.gdp_growth > 2.0:
            # Strong growth typically benefits governing parties
            factor[predictions.argmax()] *= 1.1
            factor[predictions.argmin()] *= 0.9

        return predictions * factor

    def _special_party_rules(self, predictions: torch.Tensor, parties: List[str]) -> torch.Tensor:
        """Symbolic rule: Apply special rules for specific parties."""
        for i, party in enumerate(parties):
            if party == "SSW":
                # SSW typically gets 1-2 seats when participating
                if predictions[i] > 0:
                    predictions[i] = max(1, min(2, predictions[i]))
            elif party == "Others":
                # Others typically don't get seats unless through direct mandates
                predictions[i] = 0
        return predictions

    def forward(self, x: torch.Tensor, economic_data: EconomicIndicators, parties: List[str]) -> torch.Tensor:
        # Neural prediction
        predictions = self.neural_net(x)

        # Apply symbolic rules
        for rule in self.symbolic_rules:
            if rule.__name__ == '_economic_impact_rule':
                predictions = rule(predictions, economic_data)
            elif rule.__name__ in ['_five_percent_threshold', '_special_party_rules']:
                predictions = rule(predictions, parties)
            else:
                predictions = rule(predictions)

        return predictions

def train_neuro_symbolic_model(input_data: np.ndarray, economic_data: EconomicIndicators) -> NeuroSymbolicModel:
    """Train the neuro-symbolic model with progress tracking."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Training model...", total=100)

        try:
            input_size = len(input_data)
            hidden_size = config["model"].get("hidden_size", 128)
            output_size = len(DEFAULT_PARTIES)  # Number of parties
            learning_rate = config["model"].get("learning_rate", 0.001)
            epochs = config["model"].get("epochs", 1000)

            model = NeuroSymbolicModel(input_size, hidden_size, output_size)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            # Convert data to tensors
            inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            targets = torch.tensor([TOTAL_SEATS] * output_size, dtype=torch.float32).unsqueeze(0)

            # Training loop
            model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = model(inputs, economic_data, DEFAULT_PARTIES)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                progress.update(task, completed=(epoch + 1) * 100 // epochs)
                if (epoch + 1) % 100 == 0:
                    console.print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

            console.print("[green]✓ Model training completed[/green]")
            return model

        except Exception as e:
            console.print(f"[red]Error training model: {str(e)}[/red]")
            raise ModelError(f"Model training failed: {str(e)}")

def predict_seats(
    model: NeuroSymbolicModel,
    input_data: np.ndarray,
    parties: List[str],
    economic_data: EconomicIndicators
) -> Dict[str, int]:
    """Generate seat predictions with symbolic reasoning."""
    try:
        model.eval()
        with torch.no_grad():
            inputs = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
            predictions = model(inputs, economic_data, parties)

            # Convert to seat numbers
            seat_numbers = [round(p.item()) for p in predictions[0]]

            # Adjust to ensure total seats match required number
            total_predicted = sum(seat_numbers)
            if total_predicted != TOTAL_SEATS:
                adjustment = TOTAL_SEATS - total_predicted
                max_index = seat_numbers.index(max(seat_numbers))
                seat_numbers[max_index] += adjustment

            return dict(zip(parties, seat_numbers))

    except Exception as e:
        console.print(f"[red]Error predicting seats: {str(e)}[/red]")
        raise

class ConsultingAnalysis:
    """Provides consulting-oriented analysis of election predictions."""

    def __init__(self, seat_distribution: Dict[str, int], economic_data: EconomicIndicators):
        self.seat_distribution = seat_distribution
        self.economic_data = economic_data

    def find_possible_coalitions(self) -> List[Dict]:
        """Find all mathematically possible coalitions."""
        total_seats = sum(self.seat_distribution.values())
        majority_needed = total_seats // 2 + 1
        coalitions = []

        # Get parties that have seats
        parties_with_seats = [
            party for party, seats in self.seat_distribution.items()
            if seats > 0 and party != "Others"
        ]

        # Try different coalition sizes
        for size in range(2, len(parties_with_seats) + 1):
            for combination in combinations(parties_with_seats, size):
                # Check if all parties in combination are compatible
                compatible = True
                for party in combination:
                    other_parties = [p for p in combination if p != party]
                    if not all(other in COALITION_COMPATIBILITY[party] for other in other_parties):
                        compatible = False
                        break

                if compatible:
                    seats = sum(self.seat_distribution[party] for party in combination)
                    if seats >= majority_needed:
                        coalition_set = frozenset(combination)
                        coalition = {
                            "parties": list(combination),
                            "seats": seats,
                            "majority": True,
                            "type": NAMED_COALITIONS.get(coalition_set, "Custom Coalition"),
                            "feasibility": self._assess_coalition_feasibility(combination)
                        }
                        coalitions.append(coalition)

        return sorted(coalitions, key=lambda x: (-x["seats"], -len(x["parties"])))

    def _assess_coalition_feasibility(self, parties: List[str]) -> str:
        """Assess the feasibility of a coalition based on various factors."""
        # Historical compatibility
        if all(p2 in COALITION_COMPATIBILITY[p1] for p1 in parties for p2 in parties if p1 != p2):
            base_feasibility = "High"
        else:
            return "Low"

        # Size factor - too many parties reduce feasibility
        if len(parties) > 3:
            base_feasibility = "Medium"

        # Economic factor
        if self.economic_data.unemployment_rate > 7.0 or self.economic_data.inflation_rate > 5.0:
            base_feasibility = "Medium" if base_feasibility == "High" else "Low"

        return base_feasibility

    def analyze_economic_impact(self) -> Dict:
        """Analyze impact of economic factors on election results."""
        return {
            "unemployment_impact": {
                "rate": self.economic_data.unemployment_rate,
                "analysis": self._analyze_unemployment_impact()
            },
            "gdp_impact": {
                "growth": self.economic_data.gdp_growth,
                "analysis": self._analyze_gdp_impact()
            },
            "inflation_impact": {
                "rate": self.economic_data.inflation_rate,
                "analysis": self._analyze_inflation_impact()
            }
        }

    def _analyze_unemployment_impact(self) -> str:
        """Analyze the impact of unemployment on voting patterns."""
        if self.economic_data.unemployment_rate > 7.0:
            return "Critical unemployment levels may lead to significant opposition gains"
        elif self.economic_data.unemployment_rate > 5.0:
            return "Elevated unemployment may benefit opposition parties"
        else:
            return "Stable unemployment likely favors incumbent parties"

    def _analyze_gdp_impact(self) -> str:
        """Analyze the impact of GDP growth on voting patterns."""
        if self.economic_data.gdp_growth < 0:
            return "Negative growth suggests strong anti-incumbent sentiment"
        elif self.economic_data.gdp_growth < 1.0:
            return "Low growth could lead to anti-incumbent voting"
        else:
            return "Positive growth typically benefits governing parties"

    def _analyze_inflation_impact(self) -> str:
        """Analyze the impact of inflation on voting patterns."""
        if self.economic_data.inflation_rate > 5.0:
            return "High inflation likely to cause significant voter dissatisfaction"
        elif self.economic_data.inflation_rate > 2.0:
            return "Above-target inflation may drive voters to opposition parties"
        else:
            return "Stable inflation typically maintains status quo"

    def analyze_party_trends(self) -> Dict[str, str]:
        """Analyze trends for each party."""
        trends = {}
        total_seats = sum(self.seat_distribution.values())

        for party, seats in self.seat_distribution.items():
            seat_percentage = (seats / total_seats) * 100

            if party in ["CDU/CSU", "SPD"]:
                if seat_percentage < 20:
                    trends[party] = f"Historic low for {party} at {seat_percentage:.1f}%"
                elif seat_percentage > 30:
                    trends[party] = f"Strong showing for {party} at {seat_percentage:.1f}%"
            elif party in ["Grüne", "AfD"]:
                if seat_percentage > 15:
                    trends[party] = f"Significant presence of {party} at {seat_percentage:.1f}%"
            elif party == "FDP" and seat_percentage < 5:
                trends[party] = "FDP at risk of missing parliament entry threshold"

        return trends

    def generate_report(self) -> str:
        """Generate a comprehensive analysis report."""
        coalitions = self.find_possible_coalitions()
        economic_impact = self.analyze_economic_impact()
        party_trends = self.analyze_party_trends()

        report = [
            "Bundestagswahl 2025 Analysis Report",
            "================================\n",
            "1. Seat Distribution",
            "-----------------"
        ]

        # Seat distribution
        for party, seats in self.seat_distribution.items():
            report.append(f"{party}: {seats} seats ({(seats/TOTAL_SEATS)*100:.1f}%)")

        # Coalition analysis
        report.extend([
            "\n2. Viable Coalitions",
            "------------------"
        ])

        for coalition in coalitions[:5]:  # Top 5 most likely coalitions
            report.append(
                f"{' + '.join(coalition['parties'])}: {coalition['seats']} seats "
                f"({coalition['type']}, {coalition['feasibility']} feasibility)"
            )

        # Economic factors
        report.extend([
            "\n3. Economic Factors",
            "-----------------",
            f"Unemployment Rate: {self.economic_data.unemployment_rate}% - "
            f"{economic_impact['unemployment_impact']['analysis']}",
            f"GDP Growth: {self.economic_data.gdp_growth}% - "
            f"{economic_impact['gdp_impact']['analysis']}",
            f"Inflation Rate: {self.economic_data.inflation_rate}% - "
            f"{economic_impact['inflation_impact']['analysis']}"
        ])

        # Party trends
        if party_trends:
            report.extend([
                "\n4. Notable Trends",
                "---------------"
            ])
            for party, trend in party_trends.items():
                report.append(trend)

        return "\n".join(report)
def plot_results(seat_distribution: Dict[str, int], economic_data: EconomicIndicators):
    """Create interactive visualizations of the election analysis using Plotly."""
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Predicted Seat Distribution in Bundestag', 'Economic Indicators'),
        vertical_spacing=0.2
    )

    # Prepare data for seat distribution
    parties = list(seat_distribution.keys())
    seats = list(seat_distribution.values())
    party_colors = {
        'CDU/CSU': '#000000',  # Black
        'SPD': '#E3000F',      # Red
        'Grüne': '#1AA037',    # Green
        'FDP': '#FFED00',      # Yellow
        'AfD': '#0489DB',      # Blue
        'Linke': '#BE3075',    # Purple
        'SSW': '#003D8F',      # Dark Blue
        'Freie Wähler': '#FF8000',  # Orange
        'Others': '#808080'     # Gray
    }
    colors = [party_colors.get(party, '#808080') for party in parties]

    # Add seat distribution bar chart
    fig.add_trace(
        go.Bar(
            x=parties,
            y=seats,
            name='Seats',
            marker_color=colors,
            text=seats,
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Seats: %{y}<br>' +
                         'Percentage: %{customdata:.1f}%<extra></extra>',
            customdata=[(seats[i]/TOTAL_SEATS)*100 for i in range(len(seats))]
        ),
        row=1, col=1
    )

    # Add economic indicators bar chart
    indicators = ['Unemployment', 'GDP Growth', 'Inflation']
    values = [economic_data.unemployment_rate, economic_data.gdp_growth, economic_data.inflation_rate]
    indicator_colors = ['#ff9999', '#99ff99', '#ffb366']  # Softer colors

    fig.add_trace(
        go.Bar(
            x=indicators,
            y=values,
            name='Economic Indicators',
            marker_color=indicator_colors,
            text=[f'{v:.1f}%' for v in values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>' +
                         'Value: %{y:.1f}%<extra></extra>'
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=900,
        showlegend=False,
        title_text="Bundestag Election Analysis",
        title_x=0.5,
        bargap=0.2,
        paper_bgcolor='white',
        plot_bgcolor='rgba(0,0,0,0.03)'
    )

    # Update axes
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="Number of Seats", row=1, col=1)
    fig.update_yaxes(title_text="Percentage", row=2, col=1)

    # Add grid lines
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(0,0,0,0.1)')

    # Save interactive plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = f"election_analysis_{timestamp}.html"
    fig.write_html(html_file)
    console.print(f"[green]✓ Interactive analysis plots saved as {html_file}[/green]")
    console.print(f"[green]✓ Analysis plots saved as {filename}[/green]")

async def predict_bundestag_seats(include_analysis: bool = True) -> Dict:
    """Main prediction function with integrated analysis."""
    try:
        # Fetch both polling and economic data
        parties, polling_percentages = await fetch_polling_data()
        economic_data = await fetch_economic_data()

        # Prepare input data
        input_data = preprocess_data(polling_percentages)
        setattr(preprocess_data, 'economic_data', economic_data)  # Pass economic data to preprocessing

        # Create and train model
        model = train_neuro_symbolic_model(input_data, economic_data)

        # Generate predictions
        seat_distribution = predict_seats(model, input_data, parties, economic_data)

        # Create visualizations
        plot_results(seat_distribution, economic_data)

        if include_analysis:
            # Generate consulting analysis
            analysis = ConsultingAnalysis(seat_distribution, economic_data)
            report = analysis.generate_report()

            return {
                "seat_distribution": seat_distribution,
                "economic_indicators": {
                    "unemployment_rate": economic_data.unemployment_rate,
                    "gdp_growth": economic_data.gdp_growth,
                    "inflation_rate": economic_data.inflation_rate
                },
                "analysis_report": report
            }

        return {"seat_distribution": seat_distribution}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# FastAPI app
app = FastAPI(title="Enhanced Bundestag Prediction API")

@app.get("/predict")
async def predict_endpoint(include_analysis: bool = True):
    """Enhanced prediction endpoint with optional analysis."""
    return await predict_bundestag_seats(include_analysis)

@app.get("/analysis")
async def analysis_endpoint():
    """Dedicated endpoint for consulting analysis."""
    result = await predict_bundestag_seats(include_analysis=True)
    return {"analysis": result["analysis_report"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
