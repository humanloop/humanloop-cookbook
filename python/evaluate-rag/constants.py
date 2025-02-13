PROMPT_TEMPLATE = [
    {
        "role": "system",
        "content": """
Answer the question provided by the user factually. And reference the options for the answer:

Question: {{question}}

Options:
- {{option_A}}
- {{option_B}}
- {{option_C}}
- {{option_D}}
- {{option_E}}
---
Here is some retrieved information that might be helpful.
Retrieved data:
{{retrieved_data}}
---
Give you answer in 3 sections using the following format. Do not include the quotes or the brackets. Do include the "---" separators.
<chosen option verbatim>
---
<clear explanation of why the option is correct and why the other options are incorrect. keep it ELI5.>
---
<quote relevant information snippets from the retrieved data verbatim. every line here should be directly copied from the retrieved data>
""",
    }
]


TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve_knowledge",
        "description": "Looks up relevant context in a knowledge base for a given query.",
        "parameters": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The query to retrieve knowledge to help answer.",
                },
            },
            "additionalProperties": False,
        },
    },
}
