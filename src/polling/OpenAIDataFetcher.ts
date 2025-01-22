import OpenAI from 'openai';
import { DataFetcher, PollData, DEFAULT_POLLS } from '../types';

export class OpenAIDataFetcher implements DataFetcher {
    private client: OpenAI;
    private sourceUrl: string;

    constructor(apiKey: string, sourceUrl: string) {
        this.client = new OpenAI({ apiKey });
        this.sourceUrl = sourceUrl;
    }

    async fetchPollingData(): Promise<PollData[]> {
        try {
            const response = await this.client.chat.completions.create({
                model: "gpt-4",
                messages: [{
                    role: "user",
                    content: `Extract the latest polling percentages for German political parties from ${this.sourceUrl}. Return only the data in this exact JSON format: { "polls": [{ "party": "partyname", "percentage": number }] }`
                }]
            });

            const content = response.choices[0].message.content;
            if (!content) {
                throw new Error("No content in response");
            }

            const data = JSON.parse(content);
            if (!data.polls || !Array.isArray(data.polls)) {
                throw new Error("Invalid response format");
            }

            this.validatePollingData(data.polls);
            return data.polls;
        } catch (error) {
            console.error("Error fetching data with OpenAI:", error);
            console.log("Falling back to default polling data");
            return DEFAULT_POLLS;
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
