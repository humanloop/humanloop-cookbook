"""
In this example we will demonstrate how to evaluate a chat system using an AI user.

The use case will be a candidate talking to a recruiter in a specific scenario.
e.g. scenario="Initial inquiry about junior developer role at fintech".

The function `chat_with_recruiter` represents the full AI system. It takes `scenario`
as an input and iterates through turns of calling the LLM for the candidate and for the recruiter.

We assume Prompts are stored in code and Humanloop is used as a proxy to LLM providers.
This pattern can easily be modified to instead:
- Get the Prompt from Humanloop [https://humanloop.com/docs/v5/api-reference/prompts/get]
- Call the model provider(s) directly
- Log the results to Humanloop [https://humanloop.com/docs/v5/api-reference/prompts/log]
See related guide - https://humanloop.com/docs/v5/guides/prompts/log-to-a-prompt.

Running `chat_with_recruiter` will automatically instantiate files (with uncommitted versions) and logs on Humanloop.

We then demonstrate triggering an offline evaluation for a toy dataset of scenarios.

PREREQUISITES:
- You set your OpenAI API key on Humanloop (needed for the hl.prompts.call(...) method.
- You have generated a Humanloop API key and stored it in a `.env` file in the root of this project.
Do this here: https://app.humanloop.com/account/api-keys

"""
from dotenv import load_dotenv
import os
from humanloop import Humanloop

load_dotenv()
hl = Humanloop(api_key=os.getenv("HUMANLOOP_KEY"))


def chat_with_recruiter(scenario: str, **kwargs) -> str:
    """Run a full chat session between an AI candidate and an AI recruiter.

    Args:
        scenario: A string describing the scenario the candidate will follow.
    """

    # Open the conversation flow
    flow_log = hl.flows.log(
        # You can reference file by path or ID if it already exists.
        path="recruit-bot/chat-with-recruiter",
        inputs={"scenario": scenario},
        # Include in attributes any configuration you want to uniquely define the version of the flow
        flow={"attributes": {"version": "Demo"}},
    )
    candidate_messages = []
    recruiter_messages = []
    turn = 0
    output = None

    while True:
        turn += 1
        # Candidate turn
        candidate_response = hl.prompts.call(
            path="recruit-bot/candidate",
            # Link the prompt to the flow
            trace_parent_id=flow_log.id,
            inputs={"scenario": scenario},
            messages=candidate_messages,
            # Configuration for the prompt - you can also instead set this in the UI
            # https://humanloop.com/docs/v5/quickstart/create-prompt
            prompt={
              "model": "gpt-4o-mini",
              "template": [{
                  "role": "system",
                  "content": "You are a candidate talking to a recruiter in the following scenario: {{scenario}}. Begin with just posing an initial question and wait for the recruiter to respond.""",
              }]
            },
        )
        # Update the message history for the next turn
        candidate_message = candidate_response.logs[0].output_message
        assert candidate_message
        candidate_messages.append(candidate_message)
        recruiter_messages.append({"role": "user", "content": candidate_message.content})

        # Recruiter turn
        recruiter_response = hl.prompts.call(
            path="recruit-bot/recruiter",
            trace_parent_id=flow_log.id,
            messages=recruiter_messages,
            prompt={
              "model": "gpt-4o-mini",
              "template": [{
                  "role": "system",
                  "content": "You are a helpful recruiter talking to a candidate. Only at the end of the conversation, finish your response with 'Thank you, goodbye', otherwise engage in helpful dialogue.""",
              }],
            },
        )
        recruiter_message = recruiter_response.logs[0].output_message
        assert recruiter_message
        recruiter_messages.append(recruiter_message)
        candidate_messages.append({"role": "user", "content": recruiter_message.content})

        output = f"TURN:{turn}\n\nCANDIDATE:\n{candidate_message.content}\n\nRECRUITER:\n{recruiter_message.content}"
        if "Thank you, goodbye" in recruiter_message.content:  # pyright: ignore
            break

        if turn > 10:
            break

        print(output)

    # close the conversation flow
    _ = hl.flows.update_log(
        log_id=flow_log.id,
        output=output,
        trace_status="complete"
    )
    return output


if __name__ == "__main__":
    # Trigger the application, which result in files being created on Humanloop with initial logs
    _ = chat_with_recruiter("Initial conversation about junior developer role at fintech")

    # Create some toy Evaluators on Humanloop before triggering an Evaluation
    # You can also do this via our Editor UI [https://humanloop.com/docs/v5/guides/evals/llm-as-a-judge]
    _ = hl.evaluators.upsert(
        path="recruit-bot/evaluators/quality",
        spec={
            "evaluator_type": "llm",
            "arguments_type": "target_free",
            "return_type": "number",
            "prompt": {
                "model": "gpt-4o",
                "endpoint": "complete",
                "temperature": 0,
                "template": """Your goal is to rate the quality of performance of a recruiter AI bot. Score between 1 and 10.
Higher scores for helpfulness, thoughtful questions, informativeness and putting the candidate at ease while operating in the best interest of both parties. 

Here is the scenario they are discussing.
[Input]: {{ log.inputs.scenario }}
And output of conversation:
[Output]:{{ log.output }}

And the full dialogue
[Logged Dialogue]: {{ log.children }}

First provide your rationale, then on a new line finally provide a score. A single integer between 1 and 10.
"""
            }
        }
    )

    # Evaluate the model on different scenarios - you can target the whole flow, or individual Prompt steps.
    # In this case we target the whole Flow.
    # If targeting individual Prompt steps, you can also trigger evaluations from the UI [https://humanloop.com/docs/v5/guides/evals/run-evaluation]
    _ = hl.evaluations.run(
        name="Initial experiment",
        file={
            "path": "recruit-bot/chat-with-recruiter",
            "callable": chat_with_recruiter,
        },
        dataset={
            "path": "recruit-bot/scenarios",
            "datapoints": [
                {"inputs": {"scenario": "Initial conversation about junior developer role at fintech"}},
                {"inputs": {"scenario": "Checking in on the status your application for a director role at a bank"}},
                {"inputs": {"scenario": "Asking for feedback about a recent interview rejection for a senior data scientist role"}},
            ]
        },
        # Use evaluators managed on Humanloop (NB: you can also run your Evaluators in code and log the results)
        evaluators=[{"path": "recruit-bot/evaluators/quality"}],
    )






