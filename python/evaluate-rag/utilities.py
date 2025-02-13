import os
import pandas as pd
from chromadb import chromadb


def populate_knowledge_base(chroma: chromadb.Client):
    collection = chroma.get_or_create_collection(name="MedQA")
    knowledge_base = pd.read_parquet("./assets/textbooks.parquet")
    knowledge_base = knowledge_base.sample(10, random_state=42)
    collection.add(
        documents=knowledge_base["contents"].to_list(),
        ids=knowledge_base["id"].to_list(),
    )


def read_evaluation_dataset(number_of_datapoints: int = 20):
    df = pd.read_json("./assets/datapoints.jsonl", lines=True)

    datapoints = [row.to_dict() for _, row in df.iterrows()][0:number_of_datapoints]
    return datapoints


def compute_workspace_path(path: str) -> str:
    directory = os.getenv("DIRECTORY")
    if directory:
        return f"{directory}/{path}"
    return path
