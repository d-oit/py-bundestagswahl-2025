import axios from 'axios';
import { DataFetcher, PollData } from '../types';

export class DawumDataFetcher implements DataFetcher {
    private readonly DAWUM_API = 'https://api.dawum.de/';

    async fetchPollingData(): Promise<PollData[]> {
        try {
            const response = await axios.get(this.DAWUM_API);
            const data = response.data;

            if (!data || !data.Surveys) {
                throw new Error("Invalid response from Dawum API");
            }

            // Get the latest survey (highest ID)
            const latestSurveyId = Math.max(...Object.keys(data.Surveys).map(Number));
            const latestSurvey = data.Surveys[latestSurveyId];

            if (!latestSurvey || !latestSurvey.Results) {
                throw new Error("No results in latest survey");
            }

            // Map party IDs to names using the Parties dictionary
            const parties = data.Parties;
            const polls: PollData[] = Object.entries(latestSurvey.Results).map(([partyId, percentage]) => ({
                party: parties[partyId]?.Name || `Unknown (${partyId})`,
                percentage: Number(percentage)
            }));

            this.validatePollingData(polls);
            return polls;
        } catch (error) {
            console.error("Error fetching data from Dawum:", error);
            throw error;
        }
    }

    private validatePollingData(polls: PollData[]): void {
        if (!polls.length) {
            throw new Error("Empty polling data received");
        }

        const total = polls.reduce((sum, poll) => sum + poll.percentage, 0);
        if (total < 99 || total > 101) {
            throw new Error(`Total percentage (${total}%) is not close to 100%`);
        }

        for (const poll of polls) {
            if (typeof poll.party !== 'string' || poll.party.trim() === '') {
                throw new Error("Invalid party name");
            }
            if (typeof poll.percentage !== 'number' || poll.percentage < 0 || poll.percentage > 100) {
                throw new Error(`Invalid percentage for party ${poll.party}`);
            }
        }
    }
}
