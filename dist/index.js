"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const dotenv_1 = __importDefault(require("dotenv"));
const MistralDataFetcher_1 = require("./polling/MistralDataFetcher");
const OpenAIDataFetcher_1 = require("./polling/OpenAIDataFetcher");
const DawumDataFetcher_1 = require("./polling/DawumDataFetcher");
const types_1 = require("./types");
const fs = __importStar(require("fs"));
const path = __importStar(require("path"));
dotenv_1.default.config();
function loadConfig() {
    return __awaiter(this, void 0, void 0, function* () {
        try {
            const configPath = path.join(__dirname, '../config.json');
            const configContent = fs.readFileSync(configPath, 'utf-8');
            return JSON.parse(configContent);
        }
        catch (error) {
            console.error('Error loading config:', error);
            return {
                pollingUrl: 'https://www.wahlrecht.de/umfragen/',
                dawumApiUrl: 'https://api.dawum.de/'
            };
        }
    });
}
function main() {
    return __awaiter(this, void 0, void 0, function* () {
        const config = yield loadConfig();
        const mistralKey = process.env.MISTRAL_API_KEY;
        const openaiKey = process.env.OPENAI_API_KEY;
        // Primary data source: Dawum
        const dawumFetcher = new DawumDataFetcher_1.DawumDataFetcher();
        let pollData;
        try {
            console.log('Fetching data from Dawum...');
            pollData = yield dawumFetcher.fetchPollingData();
        }
        catch (error) {
            console.error('Error with Dawum API, falling back to AI extraction...');
            // Try Mistral first, then OpenAI as fallback
            if (mistralKey) {
                try {
                    console.log('Trying Mistral AI...');
                    const mistralFetcher = new MistralDataFetcher_1.MistralDataFetcher(mistralKey, config.pollingUrl);
                    pollData = yield mistralFetcher.fetchPollingData();
                }
                catch (error) {
                    console.error('Mistral AI failed, trying OpenAI...');
                    if (!openaiKey) {
                        throw new Error('No OpenAI API key available as fallback');
                    }
                    const openAiFetcher = new OpenAIDataFetcher_1.OpenAIDataFetcher(openaiKey, config.pollingUrl);
                    pollData = yield openAiFetcher.fetchPollingData();
                }
            }
            else if (openaiKey) {
                console.log('Using OpenAI...');
                const openAiFetcher = new OpenAIDataFetcher_1.OpenAIDataFetcher(openaiKey, config.pollingUrl);
                pollData = yield openAiFetcher.fetchPollingData();
            }
            else {
                throw new Error('No AI API keys available');
            }
        }
        // Calculate seat distribution
        const totalPercentage = pollData.reduce((sum, poll) => sum + poll.percentage, 0);
        const seatDistribution = pollData.map(poll => ({
            party: poll.party,
            seats: Math.round((poll.percentage / totalPercentage) * types_1.TOTAL_SEATS)
        }));
        // Adjust for rounding errors
        const totalSeats = seatDistribution.reduce((sum, dist) => sum + dist.seats, 0);
        const diff = types_1.TOTAL_SEATS - totalSeats;
        if (diff !== 0) {
            const largest = seatDistribution.reduce((prev, curr) => prev.seats > curr.seats ? prev : curr);
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
        const majorityNeeded = Math.ceil(types_1.TOTAL_SEATS / 2);
        console.log(`\nAnalysis:`);
        console.log(`Strongest Party: ${winner.party} with ${winner.seats} seats`);
        if (winner.seats >= majorityNeeded) {
            console.log(`${winner.party} has an absolute majority`);
        }
        else {
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
    });
}
main().catch(console.error);
