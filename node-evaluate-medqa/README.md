# node-evaluate-medqa

This demo project shows you to evaluate a RAG app for medical Q&A. It uses the Typescript SDK's `evaluations.run` utility to benchmark the performance of the RAG app on a slice of the MedQA dataset and with two evaluators: `Levenshtein Distance` and `Exact Match`.

## Setup

* The project relies on a local instance of the Chroma vector database server running. You can use Docker as instructed in the code comments or provide your own instance if preferred.

* Clone the cookbook and `cd` into this directory.

* Run `yarn` in the project folder to install dependencies.

* Add your **HUMANLOOP_API_KEY** and **OPENAI_API_KEY** in an `.env` file in the same directory.

## Running

* There are three different setups you can now try, depending on how you've integrated or plan to integrate Humanloop in your project:

  * `yarn callables` is the one with the least setup: the evaluated file is defined by function living in your codebase. The dataset and Evaluators are also present locally. This is a good example if you want to evaluate your AI project with minimal changes or Humanloop setup.

  * `yarn utilities` The callable passed to `run()` is instrumented through an SDK logging utility.

  * `yarn mixed` If your project has integrated Humanloop through explicit logging calls to API, `evaluations.run(...)` will not create logs twice.