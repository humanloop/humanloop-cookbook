# node-instrumented-chat-agent

This demo project is a basic chat agent project instrumented with Humanloop. The instrumentation is done through the Typescript SDK decorators, which version and log to Files in your workspace automatically.

## Setup

* Run `yarn` in the project folder.

* Add your **HUMANLOOP_API_KEY** and **OPENAI_API_KEY** in an `.env` file in the same directory.

## Running

* The chat agent can be run in two modes

    * `yarn start` You can now have a conversation with the agent.

    * `yarn start-decorators` Same agent but instrumented with Humanloop's File decorators. They automatically version the corresponding Files and turn calls to the decorated functions into Logs.

* Don't forget to type `exit` when you're done. The conversation will be available on Humanloop in the `Chat Agent` folder.