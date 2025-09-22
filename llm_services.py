import json
from typing import List
import os
import re

from openai import OpenAI
from pydantic import ValidationError

from utils import RawNode, LLMClassificationResponse, GraphEdge, LLMGraphResponse

# --- Prompt Loading ---
PROMPT_DIR = os.path.join(os.path.dirname(__file__), "prompts")

def load_prompt_template(filename: str) -> str:
    """Loads a prompt template from the prompts directory."""
    with open(os.path.join(PROMPT_DIR, filename), 'r', encoding='utf-8') as f:
        return f.read()

PHASE1_TEMPLATE = load_prompt_template("phase1_segmentation.txt")
PHASE2_TEMPLATE = load_prompt_template("phase2_graph_inference.txt")

# --- NEW HELPER FUNCTION ---
def extract_json_from_response(text: str) -> str:
    """
    Finds and extracts the first valid JSON object from a string that might
    contain extraneous text (like a model's chain of thought).
    """
    # Find the start of the JSON object
    start_brace = text.find('{')
    if start_brace == -1:
        return ""

    # Find the end of the JSON object by matching braces
    end_brace = -1
    brace_count = 0
    for i in range(start_brace, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1

        if brace_count == 0:
            end_brace = i
            break

    if end_brace == -1:
        return ""

    return text[start_brace:end_brace + 1]


# --- LLM Services ---

def classify_and_extract_raw_nodes(
    text_chunks: List[str], client: OpenAI, model: str
) -> List[RawNode]:
    """
    Phase 1: Takes raw text chunks and uses an LLM to classify them
    into structured RawNode objects using a few-shot prompt.
    """
    chunk_count = len(text_chunks)
    formatted_chunks = "\n".join([f'CHUNK {i}:\n---\n{chunk}\n---' for i, chunk in enumerate(text_chunks)])

    user_prompt = PHASE1_TEMPLATE.format(
        chunk_count=chunk_count, formatted_chunks=formatted_chunks
    )

    print(f"Phase 1: Sending {chunk_count} chunks to LLM for classification...")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    response_content = completion.choices[0].message.content

    # Use the helper function to extract only the JSON part
    json_str = extract_json_from_response(response_content)
    if not json_str:
        print("ERROR: Could not find a valid JSON object in the LLM response.")
        print(f"Full response received:\n{response_content}")
        raise ValueError("No JSON object found in response.")

    try:
        validated_response = LLMClassificationResponse.model_validate_json(json_str) # Parse the extracted JSON
        if len(validated_response.nodes) != chunk_count:
            raise ValueError(f"LLM returned {len(validated_response.nodes)} nodes, expected {chunk_count}.")
        print("Phase 1: LLM classification successful and validated.")
        return validated_response.nodes
    except (ValidationError, ValueError) as e:
        print(f"ERROR: Phase 1 LLM response failed validation: {e}")
        print(f"Attempted to parse:\n{json_str}")
        raise

def infer_graph_relationships(
    nodes: List[RawNode], client: OpenAI, model: str
) -> List[GraphEdge]:
    """
    Phase 2: Takes a list of nodes and uses an "Architect" LLM to infer
    the relationships (edges) between them using a few-shot prompt.
    """
    formatted_nodes = json.dumps(
        [{"id": n.id, "type": n.node_type.value, "title": n.title} for n in nodes],
        indent=2,
    )

    user_prompt = PHASE2_TEMPLATE.format(formatted_nodes=formatted_nodes)

    print("Phase 2: Sending node list to LLM for relationship inference...")
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    response_content = completion.choices[0].message.content

    # Use the helper function here as well for consistency
    json_str = extract_json_from_response(response_content)
    if not json_str:
        print("ERROR: Could not find a valid JSON object in the Phase 2 LLM response.")
        print(f"Full response received:\n{response_content}")
        raise ValueError("No JSON object found in Phase 2 response.")

    try:
        validated_response = LLMGraphResponse.model_validate_json(json_str) # Parse the extracted JSON
        print("Phase 2: LLM edge inference successful and validated.")
        return validated_response.edges
    except ValidationError as e:
        print(f"ERROR: Phase 2 LLM response failed validation: {e}")
        print(f"Attempted to parse:\n{json_str}")
        raise
