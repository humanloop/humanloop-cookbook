import { HumanloopClient } from "humanloop";
import * as dotenv from "https://deno.land/std@0.224.0/dotenv/mod.ts";
import OpenAI from "https://deno.land/x/openai@v4.69.0/mod.ts";

type MessageType = { content: string; role: "system" | "user" | "assistant" };

const env = await dotenv.load({
  envPath: ".env",
});

const humanloop = new HumanloopClient(
  {
    apiKey: env["HUMANLOOP_API_KEY"] as string,
  },
  OpenAI
);
const openAIClient = new OpenAI({
  apiKey: env["OPENAI_API_KEY"] as string,
});

const calculator = humanloop.tool({
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
  path: "Andrei QA/TS Utilities/Calculator",
});

const callModel = async (messages: MessageType[]) =>
  humanloop.prompt({
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

      if (output.choices[0].message.tool_calls) {
        for (const toolCall of output.choices[0].message.tool_calls) {
          const toolCallArgs = JSON.parse(toolCall.function.arguments);
          const result = await calculator({
            operation: toolCallArgs.operation,
            num1: toolCallArgs.num1,
            num2: toolCallArgs.num2,
          });
          return `[TOOL CALL] ${result}`;
        }
      }

      return output.choices[0].message.content || "";
    },
    path: "TS Utilities/Call Model",
  })(undefined, messages);

const conversation = async () =>
  humanloop.flow({
    callable: async (inputs: any, messages: MessageType[]) => {
      messages.push({
        role: "system",
        content:
          "You are a groovy 80s surfer dude helping with math and science.",
      });
      while (true) {
        const userInput = prompt("You: ");
        if (userInput === "exit") {
          break;
        }
        messages.push({ role: "user", content: userInput || "" });
        const response = await callModel(messages);
        messages.push({
          role: "assistant",
          content: response,
        });
        console.log("Assistant:", response);
      }
    },
    path: "Andrei QA/TS Utilities/Conversation",
  })(undefined, []);

await conversation();
