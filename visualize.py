"""
Reinforce Neural Architecture
Copyright (C) 2025 Bhargav Patel

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
import json
import os
import sys
from graphviz import Digraph
import traceback

def visualize_graph(json_file_path, output_filename='computational_graph', output_format='png', view_graph=True, scale=1.5):
    """
    Loads a graph structure from a JSON file and visualizes it using Graphviz,
    placing nodes based on the 'position' attribute.

    Args:
        json_file_path (str): Path to the input JSON file.
        output_filename (str): Base name for the output file (without extension).
        output_format (str): Output format (e.g., 'png', 'svg', 'pdf').
        view_graph (bool): Whether to automatically open the generated graph file.
        scale (float): Scaling factor for node positions to adjust spacing.
    """
    print(f"Attempting to load graph structure from: {json_file_path}")
    try:
        with open(json_file_path, 'r') as json_file:
            graph_data = json.load(json_file)
        print(f"Successfully loaded {len(graph_data)} nodes.")
    except FileNotFoundError:
        print(f"Error: JSON file not found at '{json_file_path}'")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON file '{json_file_path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading the JSON file: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not isinstance(graph_data, list):
        print(f"Error: Expected JSON data to be a list of nodes, but got {type(graph_data)}")
        sys.exit(1)

    # Create a Graphviz graph (Digraph for directed graph)
    # Use 'neato' engine which respects the 'pos' attribute
    dot = Digraph(comment='Computational Graph', format=output_format, engine='neato')

    # Remove rankdir as we are using explicit positions
    # dot.attr(rankdir='TB')
    dot.attr('node', shape='box', style='filled', fillcolor='lightblue') # Default node style

    print("Adding nodes to the graph with explicit positions...")
    node_ids_in_graph = set()
    for node in graph_data:
        try:
            node_id = str(node['unique_id'])
            op_id = node.get('op_id', 'N/A') # Use .get for safety
            name = node.get('name', f'Node_{node_id}')
            player_id = node.get('player_id', '?')
            position = node.get('position') # Get the position list/tuple

            # --- Position Handling ---
            pos_str = None
            if isinstance(position, (list, tuple)) and len(position) == 2:
                row, col = position
                # Map [row, col] to Graphviz "x,y!". Scale for spacing.
                # x = col, y = -row (Graphviz y increases upwards)
                pos_x = col * scale
                pos_y = -row * scale
                pos_str = f"{pos_x},{pos_y}!" # The '!' makes the position mandatory
            else:
                print(f"Warning: Node {node_id} has invalid or missing 'position': {position}. Position ignored.")
            # --- End Position Handling ---

            # Create a multi-line label
            label = f"Name: {name}\nOpID: {op_id}\nPos: {position}\nPlayer: {player_id}"

            # Add node with position attribute if available
            node_attrs = {}
            if pos_str:
                node_attrs['pos'] = pos_str

            # Customize input node appearance
            if op_id == -1:
                node_attrs['shape'] = 'ellipse'
                node_attrs['fillcolor'] = 'lightgreen'
                dot.node(node_id, label, **node_attrs)
            else:
                dot.node(node_id, label, **node_attrs) # Use default style + pos

            node_ids_in_graph.add(node_id)

        except KeyError as e:
            print(f"Warning: Skipping node due to missing key {e}. Node data: {node}")
        except Exception as e:
            print(f"Warning: Skipping node {node.get('unique_id', 'UNKNOWN')} due to error: {e}")
            traceback.print_exc()


    print("Adding edges to the graph...")
    edges_added = 0
    for node in graph_data:
        try:
            target_id = str(node['unique_id'])
            if target_id not in node_ids_in_graph:
                continue # Skip if target node wasn't added successfully

            input_ids = node.get('input_ids', [])
            if not isinstance(input_ids, list):
                print(f"Warning: 'input_ids' for node {target_id} is not a list: {input_ids}. Skipping edges.")
                continue

            for source_unique_id in input_ids:
                source_id = str(source_unique_id)
                # Ensure the source node was also added successfully
                if source_id in node_ids_in_graph:
                    dot.edge(source_id, target_id)
                    edges_added += 1
                else:
                    print(f"Warning: Source node ID '{source_id}' for edge to '{target_id}' not found in added nodes. Skipping edge.")
        except KeyError as e:
            print(f"Warning: Skipping edges for node {node.get('unique_id', 'UNKNOWN')} due to missing key {e}.")
        except Exception as e:
            print(f"Warning: Skipping edges for node {node.get('unique_id', 'UNKNOWN')} due to error: {e}")

    print(f"Added {edges_added} edges.")

    # Render the graph to a file
    # Use os.path.abspath to ensure the path is absolute before rendering
    output_path_abs = os.path.abspath(output_filename)
    print(f"Rendering graph using '{dot.engine}' engine to '{output_path_abs}.{output_format}'...")
    try:
        # The render function saves the file and optionally opens it
        dot.render(output_path_abs, view=view_graph, cleanup=True) # cleanup=True removes intermediate dot file
        print(f"Graph saved successfully!")
    except Exception as e:
        print(f"\n--- Graphviz Rendering Error ---")
        print(f"An error occurred while rendering the graph: {e}")
        print("Please ensure:")
        print("1. Graphviz is installed correctly (https://graphviz.org/download/).")
        print("2. The Graphviz 'bin' directory is included in your system's PATH environment variable.")
        print("   You might need to restart your terminal or IDE after modifying the PATH.")
        print(f"3. The layout engine '{dot.engine}' is available in your Graphviz installation.")
        print("-------------------------------\n")
        traceback.print_exc()

# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    # Default path, assuming the script is in the same directory as the JSON file
    # Or use the path from your PPO agent script
    json_file_path = 'best_graph_structure.json'
    # json_file_path = os.path.join('ppo_seq2seq_graphs', 'best_graph_structure.json') # Path used in PPO_agent.py

    # Check if the default file exists
    if not os.path.exists(json_file_path):
        print(f"Default JSON file path not found: '{json_file_path}'")
        # Try the alternative name if the default isn't there
        alt_json_file_path = os.path.join('ppo_seq2seq_graphs', 'final_best_graph_structure.json')
        if os.path.exists(alt_json_file_path):
            print(f"Using alternative file path: '{alt_json_file_path}'")
            json_file_path = alt_json_file_path
        else:
             # Try looking in the ppo_seq2seq_graphs directory directly
             json_in_subdir = os.path.join('ppo_seq2seq_graphs', 'best_graph_structure.json')
             if os.path.exists(json_in_subdir):
                  print(f"Using file path: '{json_in_subdir}'")
                  json_file_path = json_in_subdir
             else:
                  print("Neither 'best_graph_structure.json' nor 'final_best_graph_structure.json' found in '.' or 'ppo_seq2seq_graphs/'.")
                  print("Please specify the correct path to your graph JSON file.")
                  sys.exit(1)


    output_base_name = os.path.splitext(os.path.basename(json_file_path))[0] # Use JSON filename as base
    output_dir = os.path.dirname(json_file_path) if os.path.dirname(json_file_path) else '.' # Handle case where JSON is in current dir
    output_file_path_base = os.path.join(output_dir, output_base_name)

    visualize_graph(
        json_file_path=json_file_path,
        output_filename=output_file_path_base, # Pass base path without extension
        output_format='png',  # Choose 'png', 'svg', 'pdf', etc.
        view_graph=True,      # Set to False if you don't want it to open automatically
        scale=5             # Adjust this scale factor to change node spacing (e.g., 1.0, 2.0)
    )
