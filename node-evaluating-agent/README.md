# node-evaluating-agent

This demo project shows you to evaluate a RAG app for medical Q&A. It uses the Typescript SDK's `evaluations.run` utility to benchmark the performance of the RAG app on a slice of the MedQA dataset and with two evaluators: `Levenshtein Distance` and `Exact Match`.

## Setup

* Run `yarn` in the project folder.

* Add your **HUMANLOOP_API_KEY** and **OPENAI_API_KEY** in an `.env` file in the same directory.

## Running

* There are three different setups you can now try:

  * `yarn callables` is the one with the least setup: the evaluated file is defined by function living in your codebase. The dataset and Evaluators are also present locally. This is a good example if you want to evaluate your AI project with minimal changes or Humanloop setup.

  * `yarn decorators` The callable passed to `run()` is wrapped in a Humanloop Flow decorator. If you've already integrated Humanloop in your project, you can pass the same function used in production to your evaluation setup.

  * `yarn mixed` If your project has integrated Humanloop through explicit logging steps, `evaluations.run(...)` will work as expected.