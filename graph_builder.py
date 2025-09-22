# graph_builder.py

import json
import re
from typing import List, Dict

import networkx as nx
from utils import RawNode, GraphEdge, NodeType

class KnowledgePlanningGraph:
    """Manages the construction and manipulation of the Knowledge Planning Graph."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_nodes(self, nodes: List[RawNode]):
        """Adds a list of RawNode objects to the graph."""
        for node in nodes:
            # The node ID in the graph is the RawNode's id
            self.graph.add_node(node.id, **node.model_dump())
        print(f"Added {len(nodes)} nodes to the graph.")

    def add_edges(self, edges: List[GraphEdge]):
        """Adds a list of GraphEdge objects to the graph."""
        for edge in edges:
            if self.graph.has_node(edge.source) and self.graph.has_node(edge.target):
                self.graph.add_edge(edge.source, edge.target, type=edge.type.value)
            else:
                print(f"Warning: Skipping edge from non-existent node {edge.source} or {edge.target}")
        print(f"Added {len(edges)} edges to the graph.")

    def export_graphml(self, filepath: str):
        """Exports the graph to a GraphML file for visualization."""
        nx.write_graphml(self.graph, filepath)
        print(f"Graph exported to {filepath}. You can open this with tools like Gephi.")

    def generate_sft_dataset(self) -> List[Dict]:
        """
        Phase 3: Traverses the graph to generate a high-context SFT dataset.
        """
        sft_records = []
        for node_id, data in self.graph.nodes(data=True):
            if data.get('node_type') == NodeType.WORKED_EXAMPLE:
                instruction, output = self._split_example_content(data.get('content', ''))
                if not instruction or not output:
                    continue

                # Find parent concepts for context by traversing incoming 'EXPLAINS' edges
                context_parts = []
                for predecessor_id in self.graph.predecessors(node_id):
                    edge_data = self.graph.get_edge_data(predecessor_id, node_id)
                    if edge_data and edge_data.get('type') == 'EXPLAINS':
                        parent_node = self.graph.nodes[predecessor_id]
                        context_parts.append(parent_node.get('content', ''))

                context = "\n\n".join(context_parts)

                sft_records.append({
                    "instruction": instruction.strip(),
                    "input": context.strip(),
                    "output": output.strip(),
                })
        print(f"Generated {len(sft_records)} SFT records from the graph.")
        return sft_records

    @staticmethod
    def _split_example_content(content: str) -> (str, str):
        """
        Splits a worked example's content into problem (instruction) and solution (output).
        This uses a simple heuristic; a more robust parser might be needed for complex cases.
        """
        # Split by markdown horizontal rule or "Solution:" heading
        match = re.search(r'\n\n(?:\*\*\*|---|___|\*\*Solution:\*\*)\n\n', content, re.IGNORECASE)
        if match:
            split_point = match.start()
            problem = content[:split_point]
            solution = content[split_point + len(match.group(0)):]
            return problem, solution
        return content, "" # Fallback if no clear separator is found
