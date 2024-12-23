import * as dotenv from "dotenv";
import * as readline from "readline/promises";

import { HumanloopClient } from "humanloop";
import OpenAI from "openai";

type MessageType = { content: string; role: "system" | "user" | "assistant" };

dotenv.config({
  path: ".env",
});

const humanloop = new HumanloopClient(
  {
    apiKey: process.env.HUMANLOOP_API_KEY || "",
  },
  // Pass modules to instrument for Prompt decorator
  OpenAI
);
const openAIClient = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const calculator = humanloop.tool({
  path: "Chat Agent/Calculator",
  callable: ({
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
  },
  version: {
    function: {
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
    },
  },
});

const callModel = (messages: MessageType[]) =>
  humanloop.prompt({
    path: "Chat Agent/Call Model",
    callable: async (inputs: any, messages: MessageType[]) => {
      const output = await openAIClient.chat.completions.create({
        model: "gpt-4o",
        temperature: 0.8,
        messages: messages,
        tools: [
          {
            type: "function",
            function: calculator.jsonSchema,
          } as OpenAI.ChatCompletionTool,
        ],
      });

      let llmResponse = "";

      // Check if the agent made a tool call
      if (output.choices[0].message.tool_calls) {
        for (const toolCall of output.choices[0].message.tool_calls) {
          const toolCallArgs = JSON.parse(toolCall.function.arguments);
          const result = await calculator(toolCallArgs);
          llmResponse = `[${toolCall.function.name}] ${result}`;
        }
      } else {
        llmResponse = output.choices[0].message.content || "";
      }

      return llmResponse;
    },
  })(undefined, messages);

const conversation = () =>
  humanloop.flow({
    path: "Chat Agent/Conversation",
    callable: async () => {
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
      while (true) {
        let userInput = await rl.question("You: ");
        if (userInput === "exit") {
          rl.close();
          break;
        }
        messages.push({ role: "user", content: userInput });

        const response = await callModel(messages);
        console.log("Assistant:", response);

        messages.push({
          role: "assistant",
          content: response,
        });
      }
    },
  })(undefined, undefined);

await conversation();
