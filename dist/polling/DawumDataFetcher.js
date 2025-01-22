"use strict";
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
exports.DawumDataFetcher = void 0;
const axios_1 = __importDefault(require("axios"));
class DawumDataFetcher {
    constructor() {
        this.DAWUM_API = 'https://api.dawum.de/';
    }
    fetchPollingData() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const response = yield axios_1.default.get(this.DAWUM_API);
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
                const polls = Object.entries(latestSurvey.Results).map(([partyId, percentage]) => {
                    var _a;
                    return ({
                        party: ((_a = parties[partyId]) === null || _a === void 0 ? void 0 : _a.Name) || `Unknown (${partyId})`,
                        percentage: Number(percentage)
                    });
                });
                this.validatePollingData(polls);
                return polls;
            }
            catch (error) {
                console.error("Error fetching data from Dawum:", error);
                throw error;
            }
        });
    }
    validatePollingData(polls) {
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
exports.DawumDataFetcher = DawumDataFetcher;
