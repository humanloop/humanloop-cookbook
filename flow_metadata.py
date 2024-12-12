import datetime
import os
import time
from humanloop import Humanloop

start_time = datetime.datetime.now()

humanloop = Humanloop(
    api_key=os.getenv("HUMANLOOP_KEY"),
    base_url="http://0.0.0.0:80/v5",
)

time.sleep(3)


flow_id = humanloop.flows.upsert(
    attributes={"foo": "bar"},
    path="Andrei QA/Test Flow",
).id


trace_id = humanloop.flows.log(
    id=flow_id,
    flow={"attributes": {"foo": "bar"}},
    metadata={"foo": "bar"},
    start_time=start_time,
).id

humanloop.prompts.log(
    trace_parent_id=trace_id,
    path="Andrei QA/Test Prompt",
    prompt={
        "model": "gpt-4o",
        "provider": "openai",
    },
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "What is the weather like today?",
        },
        {
            "role": "assistant",
            "content": "The weather is sunny.",
        },
    ],
)

humanloop.flows.update_log(
    log_id=trace_id,
    trace_status="complete",
    inputs={
        "question": "What is the weather like today?",
    },
    output="The weather is sunny.",
)
