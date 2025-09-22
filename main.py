# main.py

import os
import re
import json
import argparse

from openai import OpenAI
from dotenv import load_dotenv

from llm_services import classify_and_extract_raw_nodes, infer_graph_relationships
from graph_builder import KnowledgePlanningGraph

DEFAULT_API_BASE_URL = "http://localhost:1234/v1"
DEFAULT_MODEL = "local-model"

def chunk_document(markdown_text: str) -> list[str]:
    """Splits the document by double newlines."""
    # A simple but effective chunking strategy for Markdown
    return [chunk for chunk in re.split(r'\n\n+', markdown_text) if chunk.strip()]

def main():
    parser = argparse.ArgumentParser(description="Build a Knowledge Planning Graph from a Markdown file.")
    parser.add_argument("input_file", help="Path to the input Markdown file.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_API_BASE", DEFAULT_API_BASE_URL))
    parser.add_argument("--model", default=os.getenv("MODEL", DEFAULT_MODEL))
    args = parser.parse_args()

    load_dotenv()
    client = OpenAI(base_url=args.base_url, api_key="not-needed")

    base_filename = os.path.splitext(os.path.basename(args.input_file))[0]
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Starting KPG Generation for {args.input_file} ---")

    # 1. Read and Chunk Document
    with open(args.input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    chunks = chunk_document(text)

    # 2. Phase 1: Semantic Segmentation -> Raw Nodes
    raw_nodes = classify_and_extract_raw_nodes(chunks, client, args.model)

    # 3. Phase 2: Knowledge Graph Construction
    kpg = KnowledgePlanningGraph()
    # 3a. Add nodes to the graph
    kpg.add_nodes(raw_nodes)

    # 3b. Infer relationships (edges) and add them to the graph
    try:
        edges = infer_graph_relationships(raw_nodes, client, args.model)
        kpg.add_edges(edges)
    except Exception as e:
        print(f"Could not infer graph relationships, proceeding with a node-only graph. Error: {e}")

    # 4. Phase 3: Asset Generation from KPG
    print("--- Generating Assets from KPG ---")

    # 4a. Generate and save the SFT dataset
    sft_dataset = kpg.generate_sft_dataset()
    sft_output_path = os.path.join(output_dir, f"{base_filename}_sft.json")
    with open(sft_output_path, 'w', encoding='utf-8') as f:
        json.dump(sft_dataset, f, indent=2)
    print(f"SFT dataset saved to {sft_output_path}")

    # 4b. Export the graph itself for analysis and visualization
    graphml_output_path = os.path.join(output_dir, f"{base_filename}_kpg.graphml")
    kpg.export_graphml(graphml_output_path)

    print(f"\n--- KPG Generation Complete ---")

if __name__ == "__main__":
    main()
