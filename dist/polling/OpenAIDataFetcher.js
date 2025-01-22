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
exports.OpenAIDataFetcher = void 0;
const openai_1 = __importDefault(require("openai"));
const types_1 = require("../types");
class OpenAIDataFetcher {
    constructor(apiKey, sourceUrl) {
        this.client = new openai_1.default({ apiKey });
        this.sourceUrl = sourceUrl;
    }
    fetchPollingData() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                const response = yield this.client.chat.completions.create({
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
            }
            catch (error) {
                console.error("Error fetching data with OpenAI:", error);
                console.log("Falling back to default polling data");
                return types_1.DEFAULT_POLLS;
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
exports.OpenAIDataFetcher = OpenAIDataFetcher;
