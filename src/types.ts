export interface PollData {
    party: string;
    percentage: number;
}

export interface SeatDistribution {
    [party: string]: number;
}

export interface DataFetcher {
    fetchPollingData(): Promise<PollData[]>;
}

export const TOTAL_SEATS = 630;

export const DEFAULT_POLLS: PollData[] = [
    { party: "CDU/CSU", percentage: 30.5 },
    { party: "SPD", percentage: 20.0 },
    { party: "Grüne", percentage: 18.5 },
    { party: "FDP", percentage: 8.0 },
    { party: "AfD", percentage: 12.0 },
    { party: "Linke", percentage: 6.0 },
    { party: "Others", percentage: 5.0 }
];
