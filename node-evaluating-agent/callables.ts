import * as fs from "fs";

import { HumanloopClient } from "humanloop";
import { ChromaClient } from "chromadb";
import format from "string-template";
import pl from "nodejs-polars";
import OpenAI from "openai";
import * as dotenv from "dotenv";

import { levenshtein } from "./levenshtein.js";
import { exactMatch } from "./exact.match.js";
import { ChatCompletionMessage } from "openai/resources/index.js";

// 0: Load the environment variables
dotenv.config({
  path: "./.env",
});

// 1. Setup
let DIRECTORY = "SDK/Callables Evaluation";
if (process.env.DIRECTORY_PREFIX) {
  DIRECTORY = process.env.DIRECTORY_PREFIX + "/" + DIRECTORY;
}

// 2. Instantiate Humanloop client
const humanloop = new HumanloopClient({
  apiKey: process.env.HUMANLOOP_API_KEY as string,
});

// 3. Instantiate vector database
// Expects chromadb server to be running
// docker run -d -p 8000:8000 chromadb/chroma
const chroma = new ChromaClient({ path: "http://localhost:8000" });
const collection = await chroma.getOrCreateCollection({ name: "MedQA" });

let knowledgeBase = pl.readParquet("../assets/sources/textbooks.parquet");
knowledgeBase = knowledgeBase.sample({ n: 5, seed: 42 });
collection.add({
  documents: knowledgeBase.getColumn("contents").toArray(),
  ids: knowledgeBase.getColumn("id").toArray(),
});

// 4. Loading evaluation dataset
const datapoints = pl
  .readJSON("../assets/datapoints.jsonl", {
    format: "lines",
  })
  .rows()
  // This will return an array of tuples where each column contains the columns
  .slice(0, 3)
  .map((row) => {
    return {
      inputs: row[0],
      target: row[2],
    };
  });

// 5. Load template
const TEMPLATE = fs.readFileSync("./prompt.txt", "utf8");

// 6. Define RAG + Flow

async function retrievalTool(question: string) {
  const response = await collection.query({
    nResults: 1,
    queryTexts: question,
  });
  const retrievedDoc = response.documents[0][0];
  return retrievedDoc;
}

async function entrypoint(inputs: {
  question: string;
  option_A: string;
  option_B: string;
  option_C: string;
  option_D: string;
  option_E: string;
}): Promise<string> {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY as string });

  const retrievedData = await retrievalTool(inputs.question);

  const templateArgs = {
    ...inputs,
    retrievedData,
  };

  const messages = [
    {
      role: "user",
      content: format(TEMPLATE, templateArgs),
    },
  ];

  const chatCompletion = await openai.chat.completions.create({
    model: "gpt-4o",
    temperature: 0,
    messages: messages as ChatCompletionMessage[],
  });

  return chatCompletion.choices[0].message.content || "";
}

// 7. Run evaluation

humanloop.evaluations.run(
  // File
  {
    path: `${DIRECTORY}/MedQA Answer Flow`,
    callable: entrypoint,
    version: {
      attributes: {
        prompt: {
          model: "gpt-4o",
          environment: "evaluation",
        },
      },
    },
    type: "flow",
  },
  // Dataset
  {
    datapoints,
    path: `${DIRECTORY}/Dataset`,
  },
  // Evaluation Name
  "MedQA Evaluation TS Callables",
  // Evaluators
  [
    {
      path: `${DIRECTORY}/Levenshtein`,
      argsType: "target_required",
      returnType: "number",
      callable: levenshtein,
    },
    {
      path: `${DIRECTORY}/Exact Match`,
      argsType: "target_required",
      returnType: "boolean",
      callable: exactMatch,
    },
  ]
);
