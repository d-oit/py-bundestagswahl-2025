# Bundestagswahl 2025 Prediction

This TypeScript project predicts seat distribution in the 2025 German Bundestag election based on current polling data.

## Features

- Multiple data sources:
  - Primary: dawum.de API (direct polling data)
  - Fallback: AI-powered extraction from wahlrecht.de using either:
    - Mistral AI
    - OpenAI (fallback)
- Automatic seat distribution calculation
- Coalition possibility analysis
- Real-time updates

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set up environment variables:
```bash
cp .env.example .env
```

Then edit `.env` and add at least one API key:
- `MISTRAL_API_KEY`: Your Mistral AI API key
- `OPENAI_API_KEY`: Your OpenAI API key (fallback)

## Usage

Development mode:
```bash
npm run dev
```

Production:
```bash
npm run build
npm start
```

## How It Works

1. Fetches latest polling data from dawum.de
2. If dawum.de fails, falls back to AI extraction:
   - Tries Mistral AI first
   - Falls back to OpenAI if needed
3. Calculates seat distribution based on percentages
4. Analyzes possible coalitions
5. Identifies potential winning parties/combinations

## Output Example

```
Poll Results:
CDU/CSU: 30.5%
SPD: 20.0%
...

Predicted Seat Distribution:
CDU/CSU: 192 seats
SPD: 126 seats
...

Analysis:
Strongest Party: CDU/CSU with 192 seats
Possible two-party coalitions:
CDU/CSU + SPD: 318 seats
...
