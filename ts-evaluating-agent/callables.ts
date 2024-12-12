import { HumanloopClient } from "humanloop";
import { ChromaClient } from "chromadb";
import format from "string-template";
import pl from "nodejs-polars";
import * as fs from "fs";
import OpenAI from "openai";
import { ChatCompletionMessage } from "openai/resources";

// 1. Setup
let DIRECTORY = "SDK/Callables Evaluation";
if (process.env.DIRECTORY_PREFIX) {
  DIRECTORY = process.env.DIRECTORY_PREFIX + DIRECTORY;
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
  .map((row) => row[0]);

// 5. Load template
const TEMPLATE = fs.readFileSync("./prompt.txt", "utf8");

// 6. Define callables

async function retrievalTool(question: string) {
  const response = await collection.query({
    nResults: 1,
    queryTexts: question,
  });
  const retrievedDoc = response.documents[0][0];
  return retrievedDoc;
}

async function askModel(
  question: string,
  option_A: string,
  option_B: string,
  option_C: string,
  option_D: string,
  option_E: string
) {
  const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY as string });

  const retrievedData = await retrievalTool(question);

  const inputs = {
    question: question,
    option_A: option_A,
    option_B: option_B,
    option_C: option_C,
    option_D: option_D,
    option_E: option_E,
    retrieved_data: retrievedData,
  };
  const messages = [
    {
      role: "user",
      content: format(TEMPLATE, inputs),
    },
  ];

  const chatCompletion = await openai.chat.completions.create({
    model: "gpt-4o",
    temperature: 0,
    messages: messages as ChatCompletionMessage[],
  });
  return chatCompletion.choices[0].message.content;
}

async function entrypoint(
  question: string,
  option_A: string,
  option_B: string,
  option_C: string,
  option_D: string,
  option_E: string
) {
  return askModel(question, option_A, option_B, option_C, option_D, option_E);
}

humanloop.evaluations.run(
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
  {
    datapoints,
    path: `${DIRECTORY}/Dataset`,
  },
  "MedQA Evaluation TS Callables",
  [
    {
      path: `${DIRECTORY}/Levenshtein`,
      argsType: "target_required",
      returnType: "number",
      callable: 
    }
  ]
});
