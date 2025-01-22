import dotenv from 'dotenv';
import { MistralDataFetcher } from './polling/MistralDataFetcher';
import { OpenAIDataFetcher } from './polling/OpenAIDataFetcher';
import { DawumDataFetcher } from './polling/DawumDataFetcher';
import { PollData, TOTAL_SEATS } from './types';
import * as fs from 'fs';
import * as path from 'path';

dotenv.config();

interface Config {
    pollingUrl: string;
    dawumApiUrl: string;
}

async function loadConfig(): Promise<Config> {
    try {
        const configPath = path.join(__dirname, '../config.json');
        const configContent = fs.readFileSync(configPath, 'utf-8');
        return JSON.parse(configContent);
    } catch (error) {
        console.error('Error loading config:', error);
        return {
            pollingUrl: 'https://www.wahlrecht.de/umfragen/',
            dawumApiUrl: 'https://api.dawum.de/'
        };
    }
}

async function main() {
    const config = await loadConfig();
    const mistralKey = process.env.MISTRAL_API_KEY;
    const openaiKey = process.env.OPENAI_API_KEY;

    // Primary data source: Dawum
    const dawumFetcher = new DawumDataFetcher();
    let pollData: PollData[];

    try {
        console.log('Fetching data from Dawum...');
        pollData = await dawumFetcher.fetchPollingData();
    } catch (error) {
        console.error('Error with Dawum API, falling back to AI extraction...');

        // Try Mistral first, then OpenAI as fallback
        if (mistralKey) {
            try {
                console.log('Trying Mistral AI...');
                const mistralFetcher = new MistralDataFetcher(mistralKey, config.pollingUrl);
                pollData = await mistralFetcher.fetchPollingData();
            } catch (error) {
                console.error('Mistral AI failed, trying OpenAI...');
                if (!openaiKey) {
                    throw new Error('No OpenAI API key available as fallback');
                }
                const openAiFetcher = new OpenAIDataFetcher(openaiKey, config.pollingUrl);
                pollData = await openAiFetcher.fetchPollingData();
            }
        } else if (openaiKey) {
            console.log('Using OpenAI...');
            const openAiFetcher = new OpenAIDataFetcher(openaiKey, config.pollingUrl);
            pollData = await openAiFetcher.fetchPollingData();
        } else {
            throw new Error('No AI API keys available');
        }
    }

    // Calculate seat distribution
    const totalPercentage = pollData.reduce((sum, poll) => sum + poll.percentage, 0);
    const seatDistribution = pollData.map(poll => ({
        party: poll.party,
        seats: Math.round((poll.percentage / totalPercentage) * TOTAL_SEATS)
    }));

    // Adjust for rounding errors
    const totalSeats = seatDistribution.reduce((sum, dist) => sum + dist.seats, 0);
    const diff = TOTAL_SEATS - totalSeats;
    if (diff !== 0) {
        const largest = seatDistribution.reduce((prev, curr) => 
            prev.seats > curr.seats ? prev : curr
        );
        largest.seats += diff;
    }

    // Print results
    console.log('\nPoll Results:');
    pollData.forEach(poll => {
        console.log(`${poll.party}: ${poll.percentage.toFixed(1)}%`);
    });

    console.log('\nPredicted Seat Distribution:');
    seatDistribution.forEach(dist => {
        console.log(`${dist.party}: ${dist.seats} seats`);
    });

    // Print winning party/coalition analysis
    const sortedBySeats = [...seatDistribution].sort((a, b) => b.seats - a.seats);
    const winner = sortedBySeats[0];
    const majorityNeeded = Math.ceil(TOTAL_SEATS / 2);

    console.log(`\nAnalysis:`);
    console.log(`Strongest Party: ${winner.party} with ${winner.seats} seats`);
    
    if (winner.seats >= majorityNeeded) {
        console.log(`${winner.party} has an absolute majority`);
    } else {
        console.log(`Majority needed: ${majorityNeeded} seats`);
        console.log('Possible two-party coalitions:');
        
        for (let i = 0; i < sortedBySeats.length - 1; i++) {
            for (let j = i + 1; j < sortedBySeats.length; j++) {
                const combined = sortedBySeats[i].seats + sortedBySeats[j].seats;
                if (combined >= majorityNeeded) {
                    console.log(`${sortedBySeats[i].party} + ${sortedBySeats[j].party}: ${combined} seats`);
                }
            }
        }
    }
}

main().catch(console.error);
