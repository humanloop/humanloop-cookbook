import * as dotenv from "./$node_modules/dotenv/lib/main.js";
import * as readline from "readline/promises";

import { HumanloopClient } from "./$node_modules/humanloop/index.js";
import OpenAI from "./$node_modules/openai/index.mjs";

type MessageType = { content: string; role: "system" | "user" | "assistant" };

dotenv.config({
  path: ".env",
});

const humanloop = new HumanloopClient({
  apiKey: process.env.HUMANLOOP_API_KEY || "",
});
const openAIClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const calculator = ({
  operation,
  num1,
  num2,
}: {
  operation: string;
  num1: number;
  num2: number;
}) => {
  switch (operation) {
    case "add":
      return num1 + num2;
    case "subtract":
      return num1 - num2;
    case "multiply":
      return num1 * num2;
    case "divide":
      if (num2 === 0) {
        throw new Error("Cannot divide by zero");
      }
      return num1 / num2;
    default:
      throw new Error("Invalid operation");
  }
};

const CALCULATOR_JSON_SCHEMA = {
  name: "calculator",
  description: "Perform arithmetic operations on two numbers",
  strict: true,
  parameters: {
    type: "object",
    properties: {
      operation: {
        type: "string",
        description: "The operation to perform",
        enum: ["add", "subtract", "multiply", "divide"],
      },
      num1: {
        type: "number",
        description: "The first number",
      },
      num2: {
        type: "number",
        description: "The second number",
      },
    },
    required: ["operation", "num1", "num2"],
    additionalProperties: false,
  },
};

const callModel = async (traceId: string, messages: MessageType[]) => {
  const output = await openAIClient.chat.completions.create({
    model: "gpt-4o",
    temperature: 0.8,
    messages: messages,
    tools: [
      {
        type: "function",
        function: CALCULATOR_JSON_SCHEMA,
      } as OpenAI.ChatCompletionTool,
    ],
  });

  let llmResponse = "";

  // Check if the agent made a tool call
  if (output.choices[0].message.tool_calls) {
    for (const toolCall of output.choices[0].message.tool_calls) {
      const toolCallArgs = JSON.parse(toolCall.function.arguments);
      const result = calculator(toolCallArgs);
      // Log the tool call
      humanloop.tools.log({
        path: "Chat Agent/Calculator",
        inputs: toolCallArgs,
        output: JSON.stringify(result),
        traceParentId: traceId,
      });
      llmResponse = `[${toolCall.function.name}] ${result}`;
    }
  } else {
    llmResponse = output.choices[0].message.content || "";
  }

  // Log the model call
  await humanloop.prompts.log({
    path: "Chat Agent/Call Model",
    prompt: {
      model: "gpt-4o",
      temperature: 0.8,
      tools: [CALCULATOR_JSON_SCHEMA],
    },
    traceParentId: traceId,
    messages: [...messages, { role: "assistant", content: llmResponse }],
  });

  return llmResponse;
};

const conversation = async () => {
  const messages: MessageType[] = [
    {
      role: "system",
      content:
        "You are a groovy 80s surfer dude helping with math and science.",
    },
  ];
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
  // Create the Flow trace
  // Each conversation will have a unique trace
  const traceId = (
    await humanloop.flows.log({
      path: "Chat Agent/Conversation",
      startTime: new Date(),
    })
  ).id;
  while (true) {
    let userInput = await rl.question("You: ");
    if (userInput === "exit") {
      rl.close();
      break;
    }
    messages.push({ role: "user", content: userInput });

    const response = await callModel(traceId, messages);
    console.log("Assistant:", response);

    messages.push({
      role: "assistant",
      content: response,
    });
  }
  //   Close the Flow trace when the conversation is done
  await humanloop.flows.updateLog(traceId, {
    traceStatus: "complete",
    output: JSON.stringify(messages),
  });
};

await conversation();
