"""
In this example we will demonstrate how to evaluate a toy pipeline that generates queries
based on a user question.

It demonstrates how to:
1. Use Humanloop directories and datasets to version control a set of predicates
2. Set up a flow that uses a prompt to select relevant predicates for a question,
 and a different prompt to generate queries using the selected predicates.
3. Run an offline eval for an e2e test

PREREQUISITES:
- You set your OpenAI API key on Humanloop (needed for the hl.prompts.call(...) method.
- You have generated a Humanloop API key and stored it in a `.env` file in the root of this project.
Do this here: https://app.humanloop.com/account/api-keys

"""
from dotenv import load_dotenv
import os
import json
from humanloop import Humanloop

load_dotenv()
hl = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"))


def check_predicate(predicate: str, question: str, trace_id: str | None = None) -> dict[str, str |bool]:
    """ Given the question and predicate, determine if the predicate is relevant to the question
    :param predicate: the predicate to evaluate
    :param question: the question to evaluate the predicate against
    :param trace_id: the trace id of the flow if being called from larger app
    :return: a dict of the result of the evaluation [bool], the predicate[str] and the rationale for the evaluation[str]
    """
    # Call the planner prompt to determine if the predicate is relevant
    # NB: you can also instead manage the call to the provider yourself, by first
    # getting the prompt from Humanloop, calling the provider in your code, and then logging the results to HL.
    response = hl.prompts.call(
        path="global/module-planner",
        trace_parent_id=trace_id,
        inputs={"question": question, "predicate": predicate}
    )
    response_json = json.loads(response.logs[0].output)
    response_json["predicate"] = predicate
    return response_json


def plan_queries(tenant: str, user: str, question: str) -> list[str]:
    """For a given question tenant and user, return queries to execute in downstream systems`

    The basic flow is:
    - Given tenant and user, load predicates
    - Given the question and predicates, use the `module-planner` prompt to select relevant predicates
    - Given the question and relevant predicates, use the `planner` prompt to generate final array of queries

    Given predicates are just static pieces of text that we care about version controlling,
    we store them as a dataset on Humanloop.

    To support organising by `tenant` and `user`, we have directories on Humanloop
    for each tenant. And each row in the dataset has a `user` field (note, subdirectories
    could also be used for users instead if more convenient).

    A flow is used to log and evaluate the end-to-end application and prompts are used
    for the individual steps.

    :param question: The question provided by a user.
    :param user: The user who asked the question.
    :param tenant: The tenant who the user belongs to.

    :returns: A list of relevant queries to execute in downstream systems.
    """
    inputs = {
        "question": question,
        "user": user,
        "tenant": tenant,
    }
    # Get your tenant directory on Humanloop, this can hold custom datasets for your tenant
    tenant_dir = next(t for t in hl.directories.list() if t.path == tenant)
    # Get your files and filter for your dataset
    tenant_dir = hl.directories.get(id=tenant_dir.id)
    dataset = [f for f in tenant_dir.files if f.name == "user-predicates"][0]
    predicates = hl.datasets.get(id=dataset.id, include_datapoints=True).datapoints

    # Open the agent flow for logging to HL before we start assessing predicates
    flow_log = hl.flows.log(
        # In reality this will be your full application flow, instead of only selecting predicates
        path="global/query-planner",
        inputs=inputs,
        # Include in any configuration you want to uniquely define the version of the flow
        flow={"attributes": {"flow-version": "demo-v0.0.1", "predicates-used": dataset.version_id}},
    )

    # Run the `module-planner` prompt to select relevant predicates
    relevant_predicates = []
    for row in predicates:
        # Only assess predicates relevant for the user
        if row.inputs["user"] != user:
            continue
        relevant_predicates.append(
            check_predicate(row.inputs["predicate"], question, flow_log.id)
        )

    # Run the `planner` prompt to generate the final array of queries
    planner_response = hl.prompts.call(
        inputs={"question": question, "predicates": relevant_predicates},
        path="global/planner",
        trace_parent_id=flow_log.id
    )
    # NB: would recommend using tool calling or structured output for json outputs
    queries = planner_response.logs[0].output

    # Close the conversation flow
    _ = hl.flows.update_log(
        log_id=flow_log.id,
        output=json.dumps(queries),
        trace_status="complete"
    )
    return json.loads(queries)


if __name__ == "__main__":

    # Create a dataset of predicates for a specific tenant on Humanloop
    tenant = "JBC-manufacturing"
    dataset = hl.datasets.upsert(
        path=f"{tenant}/user-predicates",
        datapoints=[
            {"inputs": {"predicate": "You should always include a query fpr safet rating according to ACC", "user": "peter"}},
            {"inputs": {"predicate": "Ensure that user data complies with GDPR regulations in all queries", "user": "peter"}},
            {"inputs": {"predicate": "Always include a query for temperature information when providing weather data", "user": "peter"}},
            {"inputs": {"predicate": "Include a verification step for data integrity checks in each transaction", "user": "taylor"}}

        ]
    )

    question = """How would you design dealing with sensitive data that ensures all user
requests and data operations meet required compliance standards before proceeding
with any action?
"""
    # Call the pipeline to check it works and see the logs on Humanloop
    _ = plan_queries(user="peter", tenant=tenant, question=question)

    # Now lets run an offline eval against the `module-planner` step - because this is a
    # simple Prompt we can trigger this from the UI. We need a dataset and an evaluator to check the predicates.

    # Upload a toy dataset, this can also be managed in the UI [https://humanloop.com/docs/v5/guides/evals/create-dataset]
    _ = hl.datasets.upsert(
        path="global/datasets/check-predicates-tests",
        datapoints=[
            {
                "inputs": {
                    "question": "Can you provide the ACC safety rating for the 2022 Toyota Camry?",
                    "predicate": "You should always include a query for safety rating according to ACC"
                },
                "target": {"check": True}
            },
            {
                "inputs": {
                    "question": "What are the fuel efficiency ratings of the 2021 Ford Focus?",
                    "predicate": "You should always include a query for safety rating according to ACC"
                },
                "target": {"check": True}
            },
            {
                "inputs": {
                    "question": "How does the ACC rate the safety of the 2020 Nissan Altima?",
                    "predicate": "You should always include a query for safety rating according to ACC"
                },
                "target": {"check": True}
            },
            {
                "inputs": {
                    "question": "What entertainment features are available in the 2022 BMW 3 Series?",
                    "predicate": "You should always include a query for safety rating according to ACC"
                },
                "target": {"check": True}
            }
        ]
    )

    # First create a simple code based evaluator that will unpack the json response and check if the predicate is True
    # You can also do this via our Editor UI [https://humanloop.com/docs/v5/guides/evals/llm-as-a-judge]
    _ = hl.evaluators.upsert(
        path="global/evaluators/check-predicates",
        spec={
            "evaluator_type": "python",
            "arguments_type": "target_required",
            "return_type": "boolean",
            "code": """
import json
    
def check_predicate(log, target):
    prediction = json.loads(log["output"])
    check = target["check"]

    return prediction["predicate_result"] == check
    """
        }
    )

    # Now go and trigger an Evaluation for the `module-planner` step in the UI [from the UI [https://humanloop.com/docs/v5/guides/evals/run-evaluation]

    # ----------------------------------------------------------------------------------

    # Now let's run an e2e eval against the plan_queries function. Because this is a more
    # complex function that is orchestrated in code, we have to trigger the eval from code
    _ = hl.evaluations.run(
        name="Initial experiment",
        file={
            "path": "global/query-planner",
            "callable": plan_queries,
        },
        dataset={
            "path": "global/datasets/query-planner-tests",
            "datapoints": [
                {
                    "inputs": {
                        "question": "What is the fuel efficiency of the 2021 Honda Accord?",
                        "user": "peter", "tenant": "JBC-manufacturing"},
                    "target": {"queries": [
                        "You should always include a query for safety rating according to ACC"]},
                },
                {
                    "inputs": {
                        "question": "Provide the towing capacity of the 2022 Ford F-150.",
                        "user": "peter", "tenant": "JBC-manufacturing"},
                    "target": {"queries": [
                        "You should always include a query for safety rating according to ACC"]},
                },
                {
                    "inputs": {
                        "question": "Can you tell me the interior features of the 2023 BMW X5?",
                        "user": "peter", "tenant": "JBC-manufacturing"},
                    "target": {"queries": [
                        "You should always include a query for safety rating according to ACC"]},
                },
                {
                    "inputs": {
                        "question": "What are the color options for the 2022 Audi A4?",
                        "user": "peter", "tenant": "JBC-manufacturing"},
                    "target": {"queries": [
                        "You should always include a query for safety rating according to ACC"]},
                }
            ]
        },
        # Use evaluators managed on Humanloop (NB: you can also run your Evaluators in code and log the results)
        evaluators=[{"path": "Example Evaluators/AI/Semantic similarity"}],
    )






