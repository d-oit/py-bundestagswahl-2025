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
Object.defineProperty(exports, "__esModule", { value: true });
exports.MistralDataFetcher = void 0;
const types_1 = require("../types");
class MistralDataFetcher {
    constructor(apiKey, sourceUrl) {
        this.initialized = false;
        this.apiKey = apiKey;
        this.sourceUrl = sourceUrl;
    }
    initializeClient() {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.initialized) {
                const mistral = yield Promise.resolve().then(() => __importStar(require('@mistralai/mistralai')));
                this.client = new mistral.Mistral({ apiKey: this.apiKey });
                this.initialized = true;
            }
        });
    }
    fetchPollingData() {
        return __awaiter(this, void 0, void 0, function* () {
            try {
                yield this.initializeClient();
                const response = yield this.client.chat({
                    model: "mistral-large-latest",
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
                console.error("Error fetching data with Mistral:", error);
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
exports.MistralDataFetcher = MistralDataFetcher;
