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
# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from typing import List, Dict, Tuple, Any, Optional
import random
import string
import math
# from scipy import signal # Replace signal processing with PyTorch equivalents if needed
# import numpy.fft as np_fft # Replace with torch.fft
# from scipy import special # Keep for Factorial/Binomial Coeff CPU calculation if complex on GPU
import traceback
import json

# --- PyTorch Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.fft as torch_fft
# --- End PyTorch Imports ---


# --- MathNode Class (PyTorch Version - Unchanged from previous) ---
class MathNode:
    """
    Represents a mathematical operation node using PyTorch.
    Learnable parameter `learnable_param` acts as 'y' in f(x, y) for binary ops,
    or as bias/scale for specific unary ops.
    """
    def __init__(self,
                 op_id: int,
                 name: str,
                 op_info: Dict, # Contains 'arity', 'name', 'core_func', 'vectorized'
                 inputs: Optional[List['MathNode']] = None,
                 feature_dim: int = 8,
                 player_id: int = 0,
                 device: torch.device = torch.device('cpu')):

        self.op_id = op_id
        self.name = name
        self.op_info = op_info
        self.arity = op_info.get('arity', 0) # Get arity from op_info
        self.inputs = inputs or [] # List of input MathNode objects
        self.output_tensor: Optional[torch.Tensor] = None # Stores PyTorch tensor output
        self.position = None # (row, col) on the grid
        self.output_shape = None # Store runtime shape
        self.player_id = player_id
        self.unique_id = id(self) # Unique identifier for this node instance
        self.device = device

        # --- Learnable Parameter Handling (Revised) ---
        self.learnable_param: Optional[nn.Parameter] = None
        self.learnable_role: Optional[str] = None # e.g., 'operand_y', 'bias', 'scale', 'kernel'

        if self.arity == 2:
            # For binary ops, 'y' (second operand) is the learnable parameter.
            # Default shape (F, F) for matrix multiplication flexibility.
            # Core functions for element-wise ops will adapt (e.g., use diagonal).
            self.learnable_role = 'operand_y'
            param_shape = (feature_dim, feature_dim)
            # Initialize near identity for multiplication, near zero for addition? Or just random.
            initial_value = torch.randn(param_shape, device=self.device, dtype=torch.float32) * 0.1
            # torch.eye(feature_dim, device=self.device, dtype=torch.float32) # Alternative: near identity
            self.learnable_param = nn.Parameter(initial_value)
            # print(f"Node {self.name}: Arity 2, Role: {self.learnable_role}, Param Shape: {self.learnable_param.shape}") # Debug

        elif self.arity == 1 and op_info.get('learnable_param_role'):
             # Unary ops can have specific learnable params (e.g., bias, scale)
             role = op_info['learnable_param_role']
             self.learnable_role = role
             if role == 'bias':
                 param_shape = (feature_dim,)
                 initial_value = torch.zeros(param_shape, device=self.device, dtype=torch.float32)
             elif role == 'scale':
                 param_shape = (feature_dim,)
                 initial_value = torch.ones(param_shape, device=self.device, dtype=torch.float32)
             # Add other roles (e.g., 'kernel' for unary conv?) if needed
             else:
                  print(f"Warning: Unknown learnable_param_role '{role}' for unary op {self.name}. No param created.")
                  self.learnable_role = None # Reset role if unknown
                  param_shape = None

             if param_shape:
                  # Detach initial value before creating parameter
                  self.learnable_param = nn.Parameter(initial_value.detach())
                  # print(f"Node {self.name}: Arity 1, Role: {self.learnable_role}, Param Shape: {self.learnable_param.shape}") # Debug

        # Else (arity 0 or 1 without specified role), learnable_param remains None

    def add_input(self, node: 'MathNode'):
        if node not in self.inputs:
            self.inputs.append(node)

    def remove_input(self, node: 'MathNode'):
        try:
            self.inputs.remove(node)
        except ValueError:
            pass

    def get_parameters(self) -> List[nn.Parameter]:
        """Returns the learnable parameter of this node if it exists."""
        return [self.learnable_param] if self.learnable_param is not None else []

    def __repr__(self):
        input_ids = [inp.unique_id for inp in self.inputs]
        op_name_str = f"Op({self.op_id}:{self.op_info.get('name', 'N/A')})" if self.op_id != -1 else "Input"
        learn_info = ""
        if self.learnable_param is not None:
             p_shape = tuple(self.learnable_param.shape)
             learn_info = f", LRole={self.learnable_role}, P={p_shape}" # Use LRole

        return (f"MathNode(id={self.unique_id}, name={self.name}, op={op_name_str}, "
                f"pos={self.position}, player={self.player_id}, inputs={input_ids}{learn_info})")


# --- ComputationalGraph Class (PyTorch Version - Unchanged Internally) ---
class ComputationalGraph:
    """
    Represents a computational graph of MathNodes using PyTorch tensors.
    Manages node addition, connections, and forward pass execution.
    """
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.nodes: List[MathNode] = []
        self.nodes_by_id: Dict[int, MathNode] = {}
        self.input_node: Optional[MathNode] = None
        self.output_node: Optional[MathNode] = None
        self.grid: Dict[Tuple[int, int], MathNode] = {}
        self.max_row = -1
        self.max_col = -1
        self.device = device

    def add_node(self, node: MathNode, row: int, col: int):
        if self.get_node_at(row, col) is not None:
            raise ValueError(f"Position ({row},{col}) is already occupied by {self.get_node_at(row, col)}.")
        if node.device != self.device:
             print(f"Warning: Adding node with device {node.device} to graph with device {self.device}")
             # Move node's parameter to graph's device
             if node.learnable_param is not None:
                 node.learnable_param.data = node.learnable_param.data.to(self.device)
             node.device = self.device

        self.nodes.append(node)
        self.nodes_by_id[node.unique_id] = node
        node.position = (row, col)
        self.grid[(row, col)] = node
        self.max_row = max(self.max_row, row)
        self.max_col = max(self.max_col, col)

        if len(self.nodes) == 1: # First node is input
            self.input_node = node
            node.op_id = -1 # Mark as input type
            node.name = f"Input_0_P{node.player_id}"
            node.arity = 0 # Input node has arity 0
            node.learnable_param = None
            node.learnable_role = None

    # --- remove_node, get_node_at, etc. (unchanged) ---
    def remove_node(self, node: MathNode):
        """Removes a node and cleans up references."""
        if node not in self.nodes: return # Node already removed or never added

        # Remove from main lists/dicts
        self.nodes.remove(node)
        if node.unique_id in self.nodes_by_id:
            del self.nodes_by_id[node.unique_id]
        if node.position in self.grid:
            del self.grid[node.position]

        # Reset input/output node if it's this one
        if node == self.input_node: self.input_node = None
        if node == self.output_node: self.output_node = None

        # Remove connections *to* this node from its inputs (if any)
        # Not strictly necessary if forward pass checks cache, but good practice
        # (This part is tricky, remove_input is on the *target* node)

        # Remove connections *from* this node (remove this node from other nodes' input lists)
        nodes_to_check = list(self.nodes) # Iterate over remaining nodes
        for other_node in nodes_to_check:
            other_node.remove_input(node) # Safely remove if present

        node.inputs = [] # Clear its own inputs list

        # Recalculate grid boundaries (could be optimized)
        self.max_row = -1
        self.max_col = -1
        if self.grid:
            rows, cols = zip(*self.grid.keys())
            self.max_row = max(rows)
            self.max_col = max(cols)


    def get_node_at(self, row: int, col: int) -> Optional[MathNode]:
        return self.grid.get((row, col))

    def get_nodes_in_row(self, row: int) -> List[MathNode]:
        return sorted([node for (r, c), node in self.grid.items() if r == row], key=lambda n: n.position[1]) # Sort by col

    def get_node_by_id(self, unique_id: int) -> Optional[MathNode]:
        return self.nodes_by_id.get(unique_id)

    def connect_nodes(self, source_node_id: int, target_node_id: int) -> bool:
        source = self.get_node_by_id(source_node_id)
        target = self.get_node_by_id(target_node_id)
        if source and target and source != target: # Prevent self-loops explicitly
            target.add_input(source)
            return True
        elif source == target:
             print(f"Warning: Attempted self-connection for node {source_node_id}. Ignored.")
        return False

    def set_output_node(self, node: MathNode):
        if node in self.nodes:
            self.output_node = node
        else:
            # This can happen if the node was removed due to a cycle immediately after adding
            # print(f"Warning: Cannot set output node {node}. Node not found in graph (possibly removed).")
            self.output_node = None # Ensure output node is invalid


    def is_valid_dag(self) -> bool:
        """Checks if the graph is a Directed Acyclic Graph using Kahn's algorithm."""
        if not self.nodes: return True # Empty graph is a DAG

        in_degree = {node.unique_id: 0 for node in self.nodes}
        adj = {node.unique_id: [] for node in self.nodes}

        # Build adjacency list and calculate in-degrees
        for node in self.nodes:
            # Check for self-loops within inputs list (shouldn't happen if connect_nodes prevents it)
            if node in node.inputs:
                print(f"Error: Node {node.name} found in its own inputs list.")
                return False # Self-loop detected

            for inp_node in node.inputs:
                # Ensure input node exists in the graph's dictionary before adding edge
                if inp_node.unique_id in self.nodes_by_id and node.unique_id in self.nodes_by_id :
                    if node.unique_id not in adj[inp_node.unique_id]: # Avoid duplicate edges in adj list
                        adj[inp_node.unique_id].append(node.unique_id)
                    in_degree[node.unique_id] += 1
                else:
                    # This indicates inconsistency between node.inputs and self.nodes_by_id
                    print(f"Warning: Inconsistency detected during DAG check. Input node {inp_node.unique_id} or target node {node.unique_id} not in graph dict.")
                    # Depending on severity, might want to return False here
                    # return False

        # Initialize queue with nodes having in-degree 0
        queue = sorted([node_id for node_id, degree in in_degree.items() if degree == 0])
        count = 0 # Count of visited nodes

        while queue:
            u_id = queue.pop(0)
            count += 1

            # Process neighbors
            sorted_neighbors = sorted(adj.get(u_id, []))

            for v_id in sorted_neighbors:
                if v_id in in_degree:
                    in_degree[v_id] -= 1
                    if in_degree[v_id] == 0:
                        queue.append(v_id)
                else:
                    # If v_id is not in in_degree, it implies v_id wasn't in self.nodes initially? Error.
                    print(f"Error: Neighbor ID {v_id} of node {u_id} not found in in_degree map during DAG check.")
                    return False # Indicate graph inconsistency

            queue.sort() # Keep queue sorted for deterministic order (optional)

        # If count matches number of nodes, it's a DAG
        if count != len(self.nodes):
             print(f"DAG Check Failed: Processed {count} nodes, expected {len(self.nodes)}. Possible cycle or disconnected components.")
             # Debug: Print nodes with non-zero in-degree
             # remaining_nodes = {nid: deg for nid, deg in in_degree.items() if deg > 0}
             # print("Remaining nodes (potential cycle):", remaining_nodes)
             return False
        else:
             return True

    def topological_sort(self) -> List[MathNode]:
        """Performs topological sort using Kahn's algorithm."""
        if not self.nodes: return []

        in_degree = {node.unique_id: 0 for node in self.nodes}
        adj = {node.unique_id: [] for node in self.nodes}
        node_map = {node.unique_id: node for node in self.nodes} # Ensure map is current

        # Build adjacency list and calculate in-degrees
        for node in self.nodes:
            for inp_node in node.inputs:
                 # Ensure both source and target nodes exist before adding edge
                 if inp_node.unique_id in node_map and node.unique_id in node_map:
                     if node.unique_id not in adj[inp_node.unique_id]:
                         adj[inp_node.unique_id].append(node.unique_id)
                     in_degree[node.unique_id] += 1
                 else:
                     # This case should ideally not happen if graph management is correct
                     raise RuntimeError(f"Graph inconsistency during topological sort setup: Node {inp_node.unique_id} or {node.unique_id} not found in map.")

        # Initialize queue with nodes having in-degree 0, sorted for determinism
        queue = sorted([node_id for node_id, degree in in_degree.items() if degree == 0])
        topo_order_ids = []

        while queue:
            u_id = queue.pop(0)
            if u_id not in node_map:
                 raise RuntimeError(f"Graph inconsistency: Node ID {u_id} from queue not found in node_map.")
            topo_order_ids.append(u_id)

            # Process neighbors, sorted for determinism
            sorted_neighbors = sorted(adj.get(u_id, []))

            for v_id in sorted_neighbors:
                if v_id in in_degree:
                    in_degree[v_id] -= 1
                    if in_degree[v_id] == 0:
                        queue.append(v_id)
                else:
                    # This indicates a serious graph inconsistency
                     raise RuntimeError(f"Graph inconsistency: Neighbor ID {v_id} of node {u_id} not found in in_degree map.")

            queue.sort() # Maintain sorted order

        # Check if sort included all nodes
        if len(topo_order_ids) == len(self.nodes):
             # Return list of MathNode objects in topological order
             return [node_map[node_id] for node_id in topo_order_ids]
        else:
             # Cycle detected or graph is disconnected in a way that not all nodes were reached
             raise RuntimeError(f"Cycle detected or graph error during topological sort. Processed {len(topo_order_ids)} nodes, expected {len(self.nodes)}.")

    # This method belongs inside the ComputationalGraph class
    def forward_pass(self, input_tensor: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Perform forward pass using PyTorch tensors.
        Combines inputs via averaging (with broadcasting/expand for sequence and feature dims)
        before feeding to a node. Calls the node's core operation function.
        Requires input_tensor to be on the same device as the graph.
        """
        # --- Initial Checks ---
        if not self.nodes:
            print("Warning: Forward pass called on empty graph.")
            return None
        if self.input_node is None:
            print("Error: Input node not set in graph for forward pass.")
            return None
        if self.input_node.unique_id not in self.nodes_by_id:
            print(f"Error: Input node {self.input_node.unique_id} is no longer in the graph node map.")
            return None
        if self.output_node is None:
            print("Error: Output node not set in graph for forward pass.")
            return None
        if self.output_node.unique_id not in self.nodes_by_id:
            print(f"Error: Designated output node {self.output_node.name} ({self.output_node.unique_id}) is no longer in the graph.")
            self.output_node = None # Invalidate the reference
            return None

        if input_tensor.device != self.device:
            print(f"Warning: Input tensor device ({input_tensor.device}) differs from graph device ({self.device}). Moving input.")
            input_tensor = input_tensor.to(self.device)

        # --- Initialization ---
        node_outputs: Dict[int, Optional[torch.Tensor]] = {}
        final_output_value: Optional[torch.Tensor] = None

        try:
            # --- Assign Input Tensor to Cache ---
            node_outputs[self.input_node.unique_id] = input_tensor
            self.input_node.output_shape = tuple(input_tensor.shape)
            self.input_node.output_tensor = input_tensor

            # --- Get Execution Order ---
            sorted_nodes = self.topological_sort() # Can raise RuntimeError

            # --- Execute Nodes in Order ---
            for node in sorted_nodes:
                if node == self.input_node:
                    continue

                # --- Step 1: Gather Inputs for the Current Node ---
                inputs_ready = True
                input_tensors_for_node: List[torch.Tensor] = []
                all_input_shapes = []

                if not node.inputs:
                    print(f"Warning: Node {node.name} (id={node.unique_id}) has no inputs listed. Cannot evaluate.")
                    inputs_ready = False
                else:
                    for inp_node in sorted(node.inputs, key=lambda n: n.unique_id):
                        if inp_node.unique_id not in node_outputs:
                            print(f"CRITICAL Error: Input {inp_node.name} ({inp_node.unique_id}) for node {node.name} ({node.unique_id}) not found in output cache. Aborting forward pass.")
                            return None
                        cached_output = node_outputs.get(inp_node.unique_id)
                        if cached_output is None:
                            print(f"Warning: Input {inp_node.name} ({inp_node.unique_id}) for node {node.name} ({node.unique_id}) has None value. Cannot use for calculation.")
                            inputs_ready = False
                            break
                        input_tensors_for_node.append(cached_output)
                        all_input_shapes.append(cached_output.shape)

                # --- Step 2: Combine Inputs (if they were all valid) ---
                combined_input_data: Optional[torch.Tensor] = None
                if not inputs_ready:
                    node_outputs[node.unique_id] = None
                    node.output_tensor = None
                    if node == self.output_node: final_output_value = None
                    continue

                if not input_tensors_for_node:
                    print(f"Internal Error: Node {node.name} - inputs ready but no valid tensors collected.")
                    node_outputs[node.unique_id] = None
                    node.output_tensor = None
                    if node == self.output_node: final_output_value = None
                    continue
                elif len(input_tensors_for_node) == 1:
                    combined_input_data = input_tensors_for_node[0]
                else:
                    # --- Combine Multiple Inputs (Handling Seq AND Feature Dim Mismatches) ---
                    processed_inputs = []
                    target_batch_size = -1
                    target_seq_len = -1
                    target_feat_dim = -1
                    combination_possible = True

                    # First pass: Determine target dimensions and check basic compatibility
                    for i, t in enumerate(input_tensors_for_node):
                        if len(t.shape) == 3: # Expecting (Batch, Sequence, Feature)
                            current_b, current_s, current_f = t.shape
                            # Batch Size Check
                            if target_batch_size == -1: target_batch_size = current_b
                            elif current_b != target_batch_size:
                                print(f"Error: Node {node.name} received inputs with different batch sizes {all_input_shapes}. Cannot combine.")
                                combination_possible = False; break
                            # Determine Target Sequence Length (Max of S, 1)
                            if target_seq_len == -1: target_seq_len = current_s
                            else: target_seq_len = max(target_seq_len, current_s)
                            # Determine Target Feature Dimension (Max of F, 1)
                            if target_feat_dim == -1: target_feat_dim = current_f
                            else: target_feat_dim = max(target_feat_dim, current_f)
                        else:
                            print(f"Error: Node {node.name} received non-3D input tensor {t.shape} at index {i} (all shapes: {all_input_shapes}). Cannot combine.")
                            combination_possible = False; break

                    # Check if target dimensions are valid
                    if not combination_possible or target_batch_size <= 0 or target_seq_len <= 0 or target_feat_dim <= 0:
                        if combination_possible: # If dimensions were invalid but no explicit error
                            print(f"Warning: Node {node.name} - Could not determine valid target dimensions from inputs {all_input_shapes}.")
                        combination_possible = False # Mark as failed if dims invalid
                    else:
                        # Second pass: Process and expand each tensor to target shape
                        target_shape = (target_batch_size, target_seq_len, target_feat_dim)
                        for t in input_tensors_for_node:
                            b, s, f = t.shape
                            # Expand if needed, otherwise keep original
                            needs_expand = (s != target_seq_len) or (f != target_feat_dim)
                            if needs_expand:
                                # Check if expansion is valid (only from 1 to N)
                                if (s == 1 or s == target_seq_len) and (f == 1 or f == target_feat_dim):
                                    try:
                                        expanded_t = t.expand(*target_shape)
                                        processed_inputs.append(expanded_t)
                                    except RuntimeError as e:
                                        print(f"Error expanding tensor shape {t.shape} to target {target_shape} for node {node.name}: {e}. Aborting combination.")
                                        combination_possible = False; break
                                else: # Invalid expansion (e.g., F=2 to F=8, or S=5 to S=10)
                                    print(f"Error: Node {node.name} requires incompatible expansion from {t.shape} to {target_shape}. Inputs {all_input_shapes}. Cannot combine.")
                                    combination_possible = False; break
                            else:
                                # Shape is already the target shape
                                processed_inputs.append(t)

                    # Third pass: Stack and average if combination was successful
                    if combination_possible and processed_inputs:
                        try:
                            # Final shape check before stacking
                            if not all(p.shape == target_shape for p in processed_inputs):
                                print(f"Internal Error: Node {node.name} - Processed inputs have mismatched shapes {[p.shape for p in processed_inputs]} before stacking (Target: {target_shape}).")
                                combination_possible = False
                            else:
                                stacked_inputs = torch.stack(processed_inputs, dim=0)
                                combined_input_data = torch.mean(stacked_inputs, dim=0)
                        except Exception as e:
                              print(f"Error stacking/averaging processed inputs for node {node.name}: {e}. Setting combination to failed.")
                              combination_possible = False

                    # Assign Fallback if combination failed at any point
                    if not combination_possible:
                        # *** CHANGE: Instead of fallback, mark as failure ***
                        print(f"Marking Node {node.name} evaluation as failed due to incompatible input shapes: {all_input_shapes}.")
                        inputs_ready = False # Trigger failure path below
                        combined_input_data = None # Ensure no combined data is used
                        node_outputs[node.unique_id] = None # Mark output cache as None
                        node.output_tensor = None
                        if node == self.output_node: final_output_value = None
                        continue # Skip to next node
                    # --- End Combine Multiple Inputs ---

                # --- Check if combination succeeded (redundant if fallback is removed, but safe) ---
                if combined_input_data is None and inputs_ready: # Should not happen if logic above is correct
                    print(f"Internal Error: Node {node.name} - Combined input is None despite inputs being ready.")
                    node_outputs[node.unique_id] = None
                    node.output_tensor = None
                    if node == self.output_node: final_output_value = None
                    continue


                # --- Step 3: Apply the Node's Core Operation ---
                current_node_output: Optional[torch.Tensor] = None
                core_func = node.op_info.get('core_func')
                op_name = node.op_info.get('name', f'Op_{node.op_id}')

                if core_func:
                    try:
                        current_node_output = core_func(combined_input_data, node)

                        if current_node_output is not None:
                            if not isinstance(current_node_output, torch.Tensor):
                                print(f"Error: Core func {op_name} for node {node.name} did not return a Tensor (returned {type(current_node_output)}). Output set to None.")
                                current_node_output = None
                            elif torch.isnan(current_node_output).any() or torch.isinf(current_node_output).any():
                                print(f"Warning: Output from {op_name} (Node {node.name}) contains NaN/Inf. Clamping.")
                                current_node_output = torch.nan_to_num(current_node_output, nan=0.0, posinf=1e6, neginf=-1e6)
                            elif current_node_output.device != self.device:
                                current_node_output = current_node_output.to(self.device)

                            if current_node_output is not None:
                                node.output_shape = tuple(current_node_output.shape)

                    except Exception as e:
                        print(f"Error evaluating core function {op_name} for node {node.name} (ID:{node.unique_id}): {str(e)}")
                        current_node_output = None

                elif node.op_id != -1 :
                    print(f"Error: Core function not defined for op_id {node.op_id} (Name: {op_name}) node {node.name}. Output set to None.")
                    current_node_output = None

                # --- Step 4: Store the Result ---
                node_outputs[node.unique_id] = current_node_output
                node.output_tensor = current_node_output

                if node == self.output_node:
                    final_output_value = current_node_output

            # --- End Node Loop ---

        # --- Handle Errors During Execution ---
        except RuntimeError as e:
            print(f"Error during forward pass execution (likely graph structure error from topo sort): {e}")
            return None
        except Exception as e:
            print(f"Unexpected error during forward pass execution loop: {e}")
            traceback.print_exc()
            return None

        # --- Final Output Check ---
        if final_output_value is None:
            if self.output_node is not None:
                if self.output_node.unique_id not in node_outputs:
                    print(f"Warning: Final output is None. Designated output node ({self.output_node.name}, id={self.output_node.unique_id}) was not reached/evaluated.")
                elif node_outputs.get(self.output_node.unique_id) is None:
                    print(f"Warning: Final output is None because the designated output node ({self.output_node.name}) evaluation failed or its inputs were invalid.")
            else:
                print("Warning: Final output is None and no output node was set for the graph.")

        return final_output_value


    def get_parameters(self) -> List[nn.Parameter]:
        """Collects all learnable parameters from all nodes in the graph."""
        all_params = []
        for node in self.nodes:
            # Ensure the parameter actually exists and is a nn.Parameter
            if node.learnable_param is not None and isinstance(node.learnable_param, nn.Parameter):
                 all_params.append(node.learnable_param)
        return all_params

    # --- serialize_graph (Unchanged) ---
    def serialize_graph(self) -> List[Dict]:
        """Serializes graph structure. Does NOT save learnable parameters."""
        serialized_nodes = []
        if not self.nodes: return [] # Handle empty graph

        # Ensure node map is consistent with node list
        current_node_ids = {n.unique_id for n in self.nodes}
        valid_nodes = [n for n in self.nodes if n.unique_id in self.nodes_by_id] # Filter just in case

        for node in valid_nodes:
            # Filter input IDs to only include nodes currently in the graph
            valid_input_ids = [inp.unique_id for inp in node.inputs if inp.unique_id in current_node_ids]

            node_data = {
                "unique_id": node.unique_id,
                "op_id": node.op_id,
                "name": node.name,
                "position": node.position,
                "player_id": node.player_id,
                "input_ids": valid_input_ids, # Use filtered list
                "learnable_role": node.learnable_role, # Save role
                "op_info_name": node.op_info.get('name', 'N/A') # Store op name for clarity
            }
            serialized_nodes.append(node_data)
        return serialized_nodes

# --- MathSelfPlayEnv Class (Updated Ops) ---
class MathSelfPlayEnv(gym.Env):
    """
    Gym environment for self-play graph construction using PyTorch backend.
    Includes expanded set of operations based on README, adapted for f(x, y_learnable).
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # (Reward weights remain the same)
    ACCURACY_REWARD_WEIGHT = 1.0
    LOSS_PENALTY_WEIGHT = 0.5 # Penalize large losses
    EXPANSION_PENALTY = 0.01
    INVALID_MOVE_PENALTY = -1.0 # Penalty for illegal placement/cycle
    EVAL_FAILURE_PENALTY = -0.8 # Increased penalty for None/NaN/Inf output/loss
    SHAPE_MISMATCH_PENALTY = -0.6
    # NAN_INF_LOSS_PENALTY = -1.0 # Covered by EVAL_FAILURE_PENALTY
    LOSS_CALC_ERROR_PENALTY = -0.8
    UNEXPECTED_STEP_ERROR_PENALTY = -2.0
    IMPROVEMENT_BONUS_WEIGHT = 2.0
    NODE_EXISTENCE_REWARD = 0.01 # Small bonus for successfully adding a node

    def __init__(self,
                 grid_size=10,
                 max_steps=50,
                 feature_dim=8,
                 batch_size=64,
                 sequence_length=15,
                 task='addition',
                 device: Optional[torch.device] = None): # Allow specifying device
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.task = task

        # --- Setup Device ---
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        print(f"Environment using device: {self.device}")

        # --- Character Mapping (using PyTorch tensors) ---
        self.char_to_point: Optional[Dict[str, torch.Tensor]] = None
        self.PAD_CHAR = ' '
        self.UNK_CHAR = '<UNK>'
        self._initialize_char_to_point() # Create mapping

        # --- Define Operations (Expanded Set based on README) ---
        # ID: (Name, Arity, Core Function Name, Optional: {'learnable_param_role': 'bias'/'scale'/etc for unary})
        _potential_ops = {
            # --- Arithmetic (Binary, y=(F,F), use diag(y)) ---
            0: ("Add", 2, "add_core"),
            1: ("Sub", 2, "sub_core"),
            2: ("Mul", 2, "mul_core"),          # Elementwise Product
            3: ("Div", 2, "div_core"),
            # --- Algebra (Binary, y=(F,F), use diag(y)) ---
            4: ("Pow", 2, "pow_core"),          # x ^ diag(y)
            5: ("Root", 2, "root_core"),        # x ^ (1 / diag(y))
            # --- Linear Algebra (Binary, y=(F,F)) ---
            6: ("MatMul", 2, "matmul_core"),    # x @ y
            7: ("InnerProd", 2, "inner_prod_core"), # sum(x * diag(y), dim=-1) -> changes shape!
            # --- Number Theory (Binary, y=(F,F), use diag(y)) ---
            8: ("Mod", 2, "mod_core"),          # x % diag(y)
            # --- Transforms (Unary) ---
            9: ("FFT_Mag", 1, "fft_mag_core"),  # abs(fft(x))
            10: ("IFFT_Plc", 1, "ifft_core"),   # Placeholder/Identity
            # --- Functional Analysis (Binary) ---
            11: ("Conv1D", 2, "conv1d_core"),   # y reshaped to kernel
            # --- Activations (Unary, no learnable param) ---
            12: ("Tanh", 1, "tanh_core"),
            13: ("ReLU", 1, "relu_core"),
            14: ("Sigmoid", 1, "sigmoid_core"),
            15: ("Log", 1, "log_core"),         # Natural Log
            # --- Normalization (Unary) ---
            16: ("LayerNorm", 1, "layernorm_core", {'learnable_param_role': 'bias'}), # Has learnable bias
            # --- Order/Analysis (Unary, no learnable param) ---
            17: ("Supremum", 1, "supremum_core"), # Max reduction over sequence
            18: ("Infimum", 1, "infimum_core"),   # Min reduction over sequence
            19: ("Mean", 1, "mean_core"),       # Mean reduction over sequence
            # --- Geometry (Unary, with learnable param) ---
            20: ("Translate", 1, "translate_core", {'learnable_param_role': 'bias'}), # x + y_bias
            21: ("Scale", 1, "scale_core", {'learnable_param_role': 'scale'}),    # x * y_scale
        }

        self.operation_types = {} # map new_id -> name
        self.operations_impl = {} # map new_id -> full op_info dict
        _new_id_counter = 0

        # Sort by original key for consistent ID assignment if needed later
        for _, op_data in sorted(_potential_ops.items()):
            name, arity, core_func_name, *opts = op_data
            options_dict = opts[0] if opts else {}

            core_func = getattr(self, core_func_name, None)
            if core_func is None:
                 print(f"Warning: Core function '{core_func_name}' not found for op '{name}'. Skipping.")
                 continue

            new_id = _new_id_counter
            self.operation_types[new_id] = name

            op_info = {
                'name': name,
                'arity': arity,
                'core_func': core_func,
                **options_dict # Add options like 'learnable_param_role'
            }
            self.operations_impl[new_id] = op_info
            _new_id_counter += 1

        self.num_operations = len(self.operation_types)
        if self.num_operations == 0:
             raise ValueError("No valid operations were defined!")
        print(f"Initialized {self.num_operations} PyTorch operations:")
        # Print ops grouped by arity/type for readability
        for arity_target in [2, 1, 0]:
             print(f"--- Arity {arity_target} ---")
             ops_arity = {op_id: info for op_id, info in self.operations_impl.items() if info['arity'] == arity_target}
             for op_id, op_info in sorted(ops_arity.items()):
                 role = op_info.get('learnable_param_role', 'N/A')
                 print(f"  ID {op_id}: {op_info['name']} (Role: {role})")


        # --- Action Space (Remains NumPy based) ---
        self.num_placement_strategies = 5 # 0:RelInput, 1:Up, 2:Right, 3:Down, 4:Left
        self.action_space = spaces.Dict({
            'operation_id': spaces.Discrete(self.num_operations),
            'placement_strategy': spaces.Discrete(self.num_placement_strategies)
        })

        # --- Observation Space (Remains NumPy based) ---
        op_channels = self.num_operations
        input_channel = 1
        player1_channel = 1
        player2_channel = 1
        pointer_channel = 1
        total_channels = op_channels + input_channel + player1_channel + player2_channel + pointer_channel

        self.observation_space = spaces.Dict({
            'board': spaces.Box(
                low=0, high=1,
                shape=(self.grid_size, self.grid_size, total_channels),
                dtype=np.float32
            ),
            'current_player': spaces.Discrete(2, start=1),
            'steps_taken': spaces.Discrete(self.max_steps + 1)
        })
        print(f"Observation space board shape: {(self.grid_size, self.grid_size, total_channels)}")


        # --- Internal State ---
        self.graph: Optional[ComputationalGraph] = None
        self.current_player: int = 1
        self.pointer_location: Optional[Tuple[int, int]] = None
        self.last_loss: float = float('inf')
        self.steps_taken: int = 0
        self.current_inputs: Optional[torch.Tensor] = None
        self.target_outputs: Optional[torch.Tensor] = None

        # --- Loss Function ---
        self.loss_fn = nn.MSELoss()


    # --- _initialize_char_to_point, _generate_sequence_data (unchanged) ---
    def _initialize_char_to_point(self):
        """Initializes the character-to-vector mapping using PyTorch tensors."""
        self.char_to_point = {}
        if self.task == 'addition':
            chars = string.digits + '+'
        elif self.task == 'reverse':
            chars = string.ascii_lowercase + string.digits + " "
        else:
            raise ValueError(f"Unknown task: {self.task}")

        all_chars_for_map = set(list(chars)) | {self.PAD_CHAR, self.UNK_CHAR}
        self.feature_dim = max(self.feature_dim, 1) # Ensure feature dim is at least 1

        for char in sorted(list(all_chars_for_map)):
            if char == self.PAD_CHAR:
                point = torch.zeros(self.feature_dim, device=self.device, dtype=torch.float32)
            else:
                point = torch.randn(self.feature_dim, device=self.device, dtype=torch.float32) * 0.5
            self.char_to_point[char] = point
        print(f"Initialized char_to_point (Torch) for task '{self.task}' with {len(self.char_to_point)} chars (dim={self.feature_dim}) on {self.device}.")


    def _generate_sequence_data(self):
        """Generates a batch of sequence data as PyTorch tensors."""
        if self.char_to_point is None: self._initialize_char_to_point()

        input_sequences = []
        target_sequences = []
        max_len = self.sequence_length

        # Ensure UNK and PAD vectors exist
        unk_vector = self.char_to_point.get(self.UNK_CHAR)
        if unk_vector is None:
             unk_vector = torch.randn(self.feature_dim, device=self.device, dtype=torch.float32) * 0.1
             self.char_to_point[self.UNK_CHAR] = unk_vector
             print(f"Warning: UNK char vector created on the fly.")
        pad_vector = self.char_to_point.get(self.PAD_CHAR, torch.zeros(self.feature_dim, device=self.device))

        for _ in range(self.batch_size):
            if self.task == 'addition':
                max_digits = max(1, (max_len - 1) // 2)
                num_limit = 10**max_digits
                num1 = random.randrange(num_limit) # Use randrange to include 0
                num2 = random.randrange(num_limit)
                q = f"{num1}+{num2}"
                try:
                    a = str(num1 + num2)
                except OverflowError: # Handle potential large numbers if max_digits is huge
                    a = "Error" # Or some indicator
            elif self.task == 'reverse':
                chars = string.ascii_lowercase + string.digits + " "
                length = random.randint(1, max_len)
                q = "".join(random.choice(chars) for _ in range(length))
                a = q[::-1]
            else:
                 raise ValueError(f"Unknown task: {self.task}")

            # Pad/truncate sequences
            q_padded = q.ljust(max_len, self.PAD_CHAR)[:max_len]
            a_padded = a.ljust(max_len, self.PAD_CHAR)[:max_len]
            input_sequences.append(q_padded)
            target_sequences.append(a_padded)

        # Convert to tensors
        batch_input_tensors = []
        batch_target_tensors = []

        for q_seq, a_seq in zip(input_sequences, target_sequences):
            # Use .get with fallback to unk_vector for robustness
            q_tensor = torch.stack([self.char_to_point.get(c, unk_vector) for c in q_seq], dim=0)
            a_tensor = torch.stack([self.char_to_point.get(c, unk_vector) for c in a_seq], dim=0)
            batch_input_tensors.append(q_tensor)
            batch_target_tensors.append(a_tensor)

        # Stack into batch tensors
        try:
            self.current_inputs = torch.stack(batch_input_tensors, dim=0).to(self.device) # (B, S, F)
            self.target_outputs = torch.stack(batch_target_tensors, dim=0).to(self.device) # (B, S, F)
        except Exception as e:
             print(f"Error stacking tensors. Check sequence lengths and feature dims. Error: {e}")
             # Handle error: Maybe reset or raise? For now, print shapes.
             print("Input shapes:", [t.shape for t in batch_input_tensors])
             print("Target shapes:", [t.shape for t in batch_target_tensors])
             raise RuntimeError("Failed to create batch tensors.") from e

        # print(f"Generated data: Input shape {self.current_inputs.shape}, Target shape {self.target_outputs.shape} on {self.current_inputs.device}")

    # --- reset (unchanged) ---
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
             random.seed(seed)
             np.random.seed(seed)
             torch.manual_seed(seed)
             if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
             print(f"Environment reset with seed: {seed}")
        else:
             print("Environment reset without seed.")

        self.graph = ComputationalGraph(device=self.device)
        self.current_player = 1
        self.pointer_location = None
        self.last_loss = float('inf') # Reset loss to infinity
        self.steps_taken = 0

        try:
            self._generate_sequence_data()
        except Exception as e:
             print(f"CRITICAL ERROR during data generation in reset: {e}")
             # Cannot proceed without data, raise error
             raise RuntimeError("Failed to generate data during reset.") from e


        if self.current_inputs is None or self.target_outputs is None:
             raise RuntimeError("Data generation failed silently.")

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    # --- step (Logic mostly unchanged, but handles evaluation results better) ---
    def step(self, action: Dict[str, int]):
        """Take a turn: place node, connect, evaluate, calculate reward."""
        if self.graph is None: raise RuntimeError("Must reset environment first.")
        if self.current_inputs is None or self.target_outputs is None: raise RuntimeError("Data not initialized.")

        # --- Action Validation & Node Creation ---
        try:
            operation_id = int(action['operation_id'])
            placement_strategy = int(action['placement_strategy'])

            if not (0 <= operation_id < self.num_operations):
                 raise ValueError(f"Invalid operation_id: {operation_id} (Num ops: {self.num_operations})")
            if not (0 <= placement_strategy < self.num_placement_strategies):
                 raise ValueError(f"Invalid placement_strategy: {placement_strategy}")

            op_info = self.operations_impl[operation_id]
            op_name = op_info.get('name', 'UnknownOp')

        except (KeyError, ValueError, TypeError) as e:
             reward = self.INVALID_MOVE_PENALTY * 2 # Penalize bad action format severely
             terminated = False; truncated = False
             observation = self._get_observation()
             info = {'error': f'Invalid action format or value: {e}, action={action}', 'termination_reason': None, 'last_loss': self.last_loss}
             print(f"Step {self.steps_taken}: Invalid action received: {action}. Error: {e}")
             # Do not advance step/player on invalid action format
             return observation, reward, terminated, truncated, info

        # --- Initialize Step Variables ---
        terminated = False; truncated = False
        reward = 0.0
        info = {'error': '', 'termination_reason': None, 'node_added': False, 'dag_check': 'N/A', 'eval_status': 'N/A'}
        new_node: Optional[MathNode] = None
        target_row, target_col = -1, -1 # Initialize target position

        previous_pointer_location = self.pointer_location
        prev_max_row = self.graph.max_row
        prev_max_col = self.graph.max_col
        graph_nodes_before_ids = {n.unique_id for n in self.graph.nodes} # Store IDs for checking revert

        try:
            # --- 1. Determine Target Position ---
            is_first_op_move = (len(self.graph.nodes) <= 1) # Is this the first *operation* node?

            if not self.graph.nodes: # Absolutely first move -> create Input node
                target_row, target_col = 0, 0
                input_op_info = {'name': 'Input', 'arity': 0, 'core_func': None}
                input_node_obj = MathNode(op_id=-1, name="Input_0", op_info=input_op_info,
                                         feature_dim=self.feature_dim, player_id=self.current_player, device=self.device)
                self.graph.add_node(input_node_obj, target_row, target_col)
                self.pointer_location = (target_row, target_col)
                self.graph.set_output_node(input_node_obj) # Initial output is input
                is_first_op_move = True # Next node placed will be the first op node

            # Now determine position for the actual operation node being placed by the action
            if is_first_op_move:
                 # Place relative to input node at (0,0)
                 input_r, input_c = 0, 0
                 if placement_strategy == 1: target_row, target_col = input_r - 1, input_c # Up -> Invalid
                 elif placement_strategy == 2: target_row, target_col = input_r, input_c + 1 # Right
                 elif placement_strategy == 3: target_row, target_col = input_r + 1, input_c # Down
                 elif placement_strategy == 4: target_row, target_col = input_r, input_c - 1 # Left -> Invalid
                 elif placement_strategy == 0: raise ValueError("Placement strategy 0 (RelInput) invalid for first op node.")
                 else: raise ValueError(f"Unknown placement strategy: {placement_strategy}")
            else: # Not the first op move, place relative to pointer or input
                if self.pointer_location is None: raise RuntimeError("Pointer location is None after first move.")
                pr, pc = self.pointer_location
                if placement_strategy == 1: target_row, target_col = pr - 1, pc # Up
                elif placement_strategy == 2: target_row, target_col = pr, pc + 1 # Right
                elif placement_strategy == 3: target_row, target_col = pr + 1, pc # Down
                elif placement_strategy == 4: target_row, target_col = pr, pc - 1 # Left
                elif placement_strategy == 0: # Relative to input node
                     if self.graph.input_node and self.graph.input_node.position:
                          ir, ic = self.graph.input_node.position
                          potential_pos = [(ir, ic + 1), (ir + 1, ic)] # Try R, D
                          found = False
                          for r,c in potential_pos:
                              if (0 <= r < self.grid_size and 0 <= c < self.grid_size) and self.graph.get_node_at(r,c) is None:
                                   target_row, target_col = r, c; found = True; break
                          if not found: raise ValueError("Placement 0 failed: R/D spots near input occupied or off-grid.")
                     else: raise ValueError("Placement 0 failed: input node missing or has no position.")
                else: raise ValueError(f"Unknown placement strategy: {placement_strategy}")


            # --- 2. Validate Position ---
            if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size):
                raise ValueError(f"Invalid move: Position ({target_row},{target_col}) off-grid ({self.grid_size}x{self.grid_size}).")
            existing_node = self.graph.get_node_at(target_row, target_col)
            if existing_node is not None:
                raise ValueError(f"Invalid move: Position ({target_row},{target_col}) occupied by {existing_node.name}.")

            # --- 3. Create and Add Operation Node ---
            node_name = f"{op_name}_{len(self.graph.nodes)}_P{self.current_player}"
            new_node = MathNode(
                op_id=operation_id, name=node_name, op_info=op_info,
                feature_dim=self.feature_dim, player_id=self.current_player, device=self.device
            )
            self.graph.add_node(new_node, target_row, target_col)
            info['node_added'] = True

            # --- 4. Connect Node ---
            # Connect FROM row above
            if target_row > 0:
                sources_above = self.graph.get_nodes_in_row(target_row - 1)
                for source_node in sources_above:
                    self.graph.connect_nodes(source_node.unique_id, new_node.unique_id)
            # Connect FROM Input node if appropriate
            if self.graph.input_node and new_node != self.graph.input_node:
                needs_input_connection = (target_row == 0) or \
                                         (target_row == 1 and len(self.graph.get_nodes_in_row(0)) <= 1) # Row 0 empty or just Input
                if needs_input_connection:
                    self.graph.connect_nodes(self.graph.input_node.unique_id, new_node.unique_id)
            # Connect TO row below
            retro_connections_made = []
            next_row_idx = target_row + 1
            if next_row_idx <= self.graph.max_row:
                targets_below = self.graph.get_nodes_in_row(next_row_idx)
                for target_node in targets_below:
                    if target_node != self.graph.input_node:
                         if self.graph.connect_nodes(new_node.unique_id, target_node.unique_id):
                             retro_connections_made.append((new_node, target_node))


            # --- 5. Update Pointer ---
            self.pointer_location = (target_row, target_col)

            # --- 6. Check DAG ---
            is_dag = self.graph.is_valid_dag()
            info['dag_check'] = is_dag
            if not is_dag:
                # Revert: Remove the added node. remove_node should handle connections.
                self.graph.remove_node(new_node)
                self.pointer_location = previous_pointer_location # Restore pointer
                # Recalculate grid boundaries after removal (done within remove_node now)
                info['node_added'] = False # Mark removal
                raise ValueError("Invalid move: Created a cycle.")

            # --- Node successfully added ---
            reward += self.NODE_EXISTENCE_REWARD

            # --- 7. Evaluate Graph & Calculate Reward ---
            self.graph.set_output_node(new_node) # Evaluate with the new node as output

            current_loss = float('inf')
            eval_output: Optional[torch.Tensor] = None
            reward_component_accuracy = 0.0
            reward_component_penalty = 0.0

            if self.graph.input_node and self.graph.output_node:
                 inputs_detached = self.current_inputs.detach().clone() # Clone for safety
                 targets_detached = self.target_outputs.detach().clone()

                 # --- Perform Forward Pass ---
                 with torch.no_grad():
                      eval_output = self.graph.forward_pass(inputs_detached)

            else:
                 info['error'] = "Graph eval skipped: Input/Output node missing or invalid post-add."
                 info['eval_status'] = 'Skipped'
                 reward_component_penalty += self.EVAL_FAILURE_PENALTY # Penalize failure to evaluate

            # --- Calculate Loss and Reward based on eval_output ---
            if eval_output is not None:
                 # Check for NaN/Inf in output tensor itself
                 if torch.isnan(eval_output).any() or torch.isinf(eval_output).any():
                      info['error'] = "Evaluation forward pass resulted in NaN/Inf tensor."
                      info['eval_status'] = 'NaN/Inf Output'
                      current_loss = float('inf') # Treat as max loss
                      reward_component_penalty += self.EVAL_FAILURE_PENALTY
                 # Check shape match BEFORE calculating loss
                 elif eval_output.shape != targets_detached.shape:
                      info['error'] = f"Output shape mismatch: Exp {targets_detached.shape}, Got {eval_output.shape}"
                      info['eval_status'] = 'Shape Mismatch'
                      current_loss = float('inf') # Treat as max loss
                      reward_component_penalty += self.SHAPE_MISMATCH_PENALTY
                 else: # Shapes match, proceed to loss calculation
                     try:
                         loss = self.loss_fn(eval_output, targets_detached)
                         loss_item = loss.item()

                         if not np.isfinite(loss_item):
                             current_loss = float('inf') # Treat as max loss
                             reward_component_penalty += self.EVAL_FAILURE_PENALTY # Use general failure penalty
                             info['error'] = f"Loss calculation resulted in non-finite value: {loss_item}"
                             info['eval_status'] = 'NaN/Inf Loss'
                         else:
                             current_loss = loss_item
                             info['eval_status'] = 'Success'
                             # Calculate improvement reward (handle initial infinite last_loss)
                             finite_last_loss = self.last_loss if np.isfinite(self.last_loss) else current_loss + 1.0 # Compare against slightly worse
                             improvement = finite_last_loss - current_loss
                             # Reward is based on improvement AND scaled negative loss
                             reward_component_accuracy = (self.IMPROVEMENT_BONUS_WEIGHT * improvement) - \
                                                         (self.LOSS_PENALTY_WEIGHT * current_loss)

                     except Exception as loss_calc_e:
                          current_loss = float('inf') # Treat as max loss
                          reward_component_penalty += self.LOSS_CALC_ERROR_PENALTY
                          info['error'] = f"Error during loss calculation: {loss_calc_e}"
                          info['eval_status'] = 'Loss Error'
                          # traceback.print_exc()

            elif not info['error']: # Eval failed (forward_pass returned None)
                 current_loss = float('inf') # Treat as max loss
                 reward_component_penalty += self.EVAL_FAILURE_PENALTY
                 info['error'] = "Graph evaluation failed: forward_pass returned None."
                 info['eval_status'] = 'Forward Pass Failed'

            # Expansion penalty
            expanded_grid = (self.graph.max_row > prev_max_row) or (self.graph.max_col > prev_max_col)
            if expanded_grid:
                 reward_component_penalty -= self.EXPANSION_PENALTY
                 # info['grid_expanded'] = True # Less critical info

            # Combine reward components
            reward += reward_component_accuracy + reward_component_penalty

            # Update last_loss only if current evaluation was successful and loss is finite
            if info['eval_status'] == 'Success' and np.isfinite(current_loss):
                 self.last_loss = current_loss
            # --- End Reward Calc ---

        except ValueError as e: # Catch placement/cycle/validation errors during node add/connect
            reward = self.INVALID_MOVE_PENALTY # Apply specific penalty
            info['error'] = f"Invalid Move: {str(e)}"
            self.pointer_location = previous_pointer_location # Restore pointer

            # Check if the node was added before the error and needs removal
            current_node_ids = {n.unique_id for n in self.graph.nodes}
            added_node_id = list(current_node_ids - graph_nodes_before_ids)
            if added_node_id:
                 node_to_remove = self.graph.get_node_by_id(added_node_id[0])
                 if node_to_remove:
                     # print(f"Attempting to remove node {node_to_remove.name} after error: {e}")
                     self.graph.remove_node(node_to_remove)
            info['node_added'] = False # Ensure flag is false

        except Exception as e: # Catch unexpected errors (e.g., in data generation, core funcs)
            reward = self.UNEXPECTED_STEP_ERROR_PENALTY
            info['error'] = f"Unexpected error in step: {str(e)}"
            traceback.print_exc()
            terminated = True # End episode on unexpected serious error
            info['termination_reason'] = 'unexpected_error'
            self.pointer_location = previous_pointer_location # Attempt pointer restore

        # --- 8. Update Step Counter & Player ---
        self.steps_taken += 1
        self.current_player = 3 - self.current_player

        # --- 9. Check Termination Conditions ---
        if not terminated:
            if self.steps_taken >= self.max_steps:
                truncated = True; terminated = False
                info['termination_reason'] = 'max_steps_reached'
            elif len(self.graph.nodes) >= self.grid_size * self.grid_size:
                 truncated = True
                 info['termination_reason'] = 'grid_full' # More specific reason

        # --- 10. Prepare Return Values ---
        observation = self._get_observation()
        if not np.isfinite(reward):
            print(f"Warning: Non-finite reward calculated ({reward}). Clamping to {self.UNEXPECTED_STEP_ERROR_PENALTY}.")
            reward = self.UNEXPECTED_STEP_ERROR_PENALTY
        # Report current loss state, ensuring it's finite for logging/tracking
        info['last_loss'] = self.last_loss if np.isfinite(self.last_loss) else 1e6 # Use large finite number if inf

        # Debug Print (optional)
        # print(f"Step {self.steps_taken} Ret: P{self.current_player} | Rew={reward:.3f} | Loss={info['last_loss']:.3f} | Eval={info['eval_status']} | DAG={info['dag_check']} | Err='{info['error'][:50]}...'")

        return observation, reward, terminated, truncated, info

    # --- _get_observation, _get_info (Unchanged) ---
    def _get_observation(self):
        """Construct the NumPy observation dictionary for the agent."""
        board_shape = self.observation_space['board'].shape
        expected_channels = self.num_operations + 4
        if board_shape[2] != expected_channels:
            print(f"Warning: Observation shape mismatch. Expected {expected_channels} channels, shape is {board_shape}. Re-creating.")
            board_shape = (self.grid_size, self.grid_size, expected_channels)
            # Update the observation space if needed (though technically should be fixed at init)
            # self.observation_space['board'] = spaces.Box(low=0, high=1, shape=board_shape, dtype=np.float32)

        board = np.zeros(board_shape, dtype=np.float32)

        if self.graph is None: return {'board': board, 'current_player': self.current_player, 'steps_taken': self.steps_taken}

        op_channel_offset = 0
        input_channel_idx = self.num_operations
        player1_channel_idx = self.num_operations + 1
        player2_channel_idx = self.num_operations + 2
        pointer_channel_idx = self.num_operations + 3

        for node in self.graph.nodes:
            if node.position is None: continue
            r, c = node.position
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                if node.op_id == -1: # Input node
                    if input_channel_idx < board.shape[2]: board[r, c, input_channel_idx] = 1.0
                elif 0 <= node.op_id < self.num_operations: # Valid operation
                    ch_idx = op_channel_offset + node.op_id
                    if ch_idx < board.shape[2]: board[r, c, ch_idx] = 1.0

                # Player ownership
                if node.player_id == 1:
                     if player1_channel_idx < board.shape[2]: board[r, c, player1_channel_idx] = 1.0
                elif node.player_id == 2:
                     if player2_channel_idx < board.shape[2]: board[r, c, player2_channel_idx] = 1.0

        if self.pointer_location:
            pr, pc = self.pointer_location
            if 0 <= pr < self.grid_size and 0 <= pc < self.grid_size:
                 if pointer_channel_idx < board.shape[2]: board[pr, pc, pointer_channel_idx] = 1.0

        return {
            'board': board,
            'current_player': self.current_player,
            'steps_taken': self.steps_taken
        }

    def _get_info(self):
        finite_loss = self.last_loss if np.isfinite(self.last_loss) else 1e6 # Use large finite value
        return {
            'last_loss': finite_loss,
            'nodes_count': len(self.graph.nodes) if self.graph else 0,
            'pointer': self.pointer_location,
            'max_row': self.graph.max_row if self.graph else -1,
            'max_col': self.graph.max_col if self.graph else -1,
            'num_ops': self.num_operations
        }

    # --- render (Unchanged) ---
    def render(self, mode='human'):
        if mode != 'human' or self.graph is None: return
        grid_cell_width = 15 # Keep increased width
        total_width = self.grid_size * (grid_cell_width + 1) + 1
        print("\n" + "=" * total_width)
        print(f"Step: {self.steps_taken}/{self.max_steps}, Player: {self.current_player}'s Turn, Device: {self.device}")
        loss_val = self.last_loss if np.isfinite(self.last_loss) else float('inf')
        print(f"Last Eval Loss (MSE): {loss_val:.4f}")
        print(f"Pointer: {self.pointer_location}, Nodes: {len(self.graph.nodes)}/{self.grid_size**2}")

        grid_repr = [['.' * (grid_cell_width-1) + ' ' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                node = self.graph.get_node_at(r, c)
                cell_str = ""
                if node:
                    op_name_short = node.op_info.get('name', 'ERR')[:6] if node.op_id != -1 else 'Input'
                    player_mark = f"P{node.player_id}"
                    learn_info = ""
                    if node.learnable_param is not None:
                         p_shape = tuple(node.learnable_param.shape)
                         role_str = f"R:{node.learnable_role}" if node.learnable_role else "R:?"
                         shape_str = f"S:{p_shape}"
                         learn_info = f"({role_str}{shape_str})"

                    cell_str = f"{op_name_short}{player_mark}{learn_info}"

                if (r, c) == self.pointer_location: cell_str += "*"
                grid_repr[r][c] = f"{cell_str:<{grid_cell_width-1}}"[:grid_cell_width-1] + ' '

        print("\nBoard:")
        header = "  " + " ".join(f"{i:^{grid_cell_width}}" for i in range(self.grid_size))
        print(header)
        print(" " + "=" * total_width)
        for r in range(self.grid_size):
            print(f"{r}|" + "|".join(f"{cell}" for cell in grid_repr[r]) + "|")
            print(" " + "-" * total_width)

        print("\nConnections & Node Details (Target <- Sources):")
        sorted_nodes = sorted(self.graph.nodes, key=lambda n: (n.position[0], n.position[1]) if n.position else (999, 999))
        for node in sorted_nodes:
             input_names = sorted([f"{inp.name}({inp.unique_id % 1000})" for inp in node.inputs])
             learn_str = ""
             if node.learnable_param is not None:
                 p_shape = tuple(node.learnable_param.shape)
                 learn_str = f", LRole={node.learnable_role}, P={p_shape}"
             print(f"  {node.name}({node.unique_id % 1000}) @{node.position} (Op:{node.op_id}, Ar:{node.arity}{learn_str}) <- {input_names}")
        print("=" * total_width + "\n")


    # === Core Operation Functions (PyTorch - Added & Revised) ===

    # --- Helpers ---
    def _get_learnable_param_as_vector(self, node: MathNode, op_name: str) -> Optional[torch.Tensor]:
        """Safely retrieves the learnable param and attempts to return it as a vector (F,)."""
        if node.learnable_param is None:
            print(f"Error in {op_name}_core ({node.name}): learnable_param is None.")
            return None

        y_param = node.learnable_param
        feature_dim = self.feature_dim

        if y_param.shape == (feature_dim, feature_dim):
            return torch.diag(y_param) # Return diagonal
        elif y_param.shape == (feature_dim,):
            return y_param # Already a vector
        elif y_param.numel() == feature_dim:
             print(f"Warning in {op_name}_core ({node.name}): learnable_param shape {y_param.shape} not standard, attempting reshape to ({feature_dim},).")
             try:
                 return y_param.view(feature_dim)
             except RuntimeError:
                 print(f"Error: Could not reshape learnable param {y_param.shape} to vector.")
                 return None
        else:
             print(f"Error in {op_name}_core ({node.name}): learnable_param shape {y_param.shape} incompatible with vector use.")
             return None

    def _apply_elementwise_binary(self, x: torch.Tensor, node: MathNode, op_func: callable, op_name: str) -> torch.Tensor:
        """Helper for element-wise binary ops using diagonal/vector of learnable param."""
        y_vector = self._get_learnable_param_as_vector(node, op_name)
        if y_vector is None:
            return x # Fallback to input if param is invalid

        try:
            # Broadcasting: (B, S, F) op (F,) -> (B, S, F)
            return op_func(x, y_vector)
        except RuntimeError as e:
             print(f"Error during {op_name}_core ({node.name}): {e}. Shapes: x={x.shape}, y_vec={y_vector.shape}")
             return x # Fallback

    # --- Arithmetic & Basic Ops ---
    def add_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        return self._apply_elementwise_binary(x, node, torch.add, "add")

    def sub_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        return self._apply_elementwise_binary(x, node, torch.sub, "sub")

    def mul_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        return self._apply_elementwise_binary(x, node, torch.mul, "mul")

    def div_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        y_vector = self._get_learnable_param_as_vector(node, "div")
        if y_vector is None: return x
        try:
            y_safe = y_vector + 1e-8 * torch.sign(y_vector).detach()
            y_safe[y_safe == 0] = 1e-8
            return torch.div(x, y_safe)
        except RuntimeError as e:
             print(f"Error during div_core ({node.name}): {e}. Shapes: x={x.shape}, y_safe={y_safe.shape}")
             return x

    def pow_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # x ^ diag(y)
        y_vector = self._get_learnable_param_as_vector(node, "pow")
        if y_vector is None: return x
        try:
            # Use abs base? Clamp exponent? Be careful with gradients for negative bases.
            base = torch.relu(x) + 1e-6 # Ensure base is non-negative for stability
            # base = x # Alternative: allow negative base but risk NaN gradients
            exponent = torch.clamp(y_vector, -10, 10) # Clamp exponent range
            return torch.pow(base, exponent)
        except RuntimeError as e:
             print(f"Error during pow_core ({node.name}): {e}. Shapes: x={x.shape}, exponent={exponent.shape}")
             return x

    def root_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # x ^ (1 / diag(y))
        y_vector = self._get_learnable_param_as_vector(node, "root")
        if y_vector is None: return x
        try:
            # --- Stabilize the exponent denominator ---
            # Ensure non-zero by clamping absolute value away from zero, then restore sign
            y_abs_clamped = torch.clamp(torch.abs(y_vector), min=1e-6) # Clamp magnitude away from 0
            y_safe = y_abs_clamped * torch.sign(y_vector)
            # Handle case where original sign was zero (clamp introduced non-zero) - set to small value
            y_safe[y_vector == 0] = 1e-6
            # --- Calculate clamped exponent ---
            exponent = 1.0 / y_safe
            exponent = torch.clamp(exponent, -20, 20) # Clamp final exponent more aggressively

            # --- Ensure base is non-negative ---
            base = torch.relu(x) + 1e-7 # Ensure base non-negative and slightly > 0

            # --- Perform power operation ---
            result = torch.pow(base, exponent)

            # --- Final clamp on output just in case ---
            result = torch.clamp(result, -1e6, 1e6)

            return result
        except RuntimeError as e:
             print(f"Error during root_core ({node.name}): {e}. Shapes: x={x.shape}, exponent={exponent.shape if 'exponent' in locals() else 'N/A'}")
             return x # Fallback
        except Exception as e: # Catch other potential errors
             print(f"Unexpected error in root_core ({node.name}): {e}")
             return x

    def mod_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # x % diag(y)
        y_vector = self._get_learnable_param_as_vector(node, "mod")
        if y_vector is None: return x
        try:
            # Ensure divisor is positive for stability? torch.fmod handles signs.
            y_abs = torch.abs(y_vector) + 1e-6 # Use positive divisor
            return torch.fmod(x, y_abs)
        except RuntimeError as e:
             print(f"Error during mod_core ({node.name}): {e}. Shapes: x={x.shape}, y_vec={y_vector.shape}")
             return x

    # --- Linear Algebra ---
    # This method belongs inside the MathSelfPlayEnv class

    def matmul_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        """
        Performs matrix multiplication x @ y.
        Adapts if input x has feature dim 1 (e.g., from InnerProd):
        treats y as a linear layer using its first column to expand features back to F.
        y is the learnable (F, F) matrix.
        """
        op_name = "matmul_core" # For error messages
        feature_dim = self.feature_dim # F

        # --- Validate Learnable Parameter ---
        if node.learnable_param is None:
            print(f"Error in {op_name} ({node.name}): learnable_param is None.")
            return x # Fallback
        y = node.learnable_param
        if y.shape != (feature_dim, feature_dim):
            print(f"Error in {op_name} ({node.name}): Invalid learnable_param shape {y.shape}, expected {(feature_dim, feature_dim)}.")
            return x # Fallback

        # --- Validate Input Tensor Shape ---
        if x.ndim != 3:
            print(f"Error in {op_name} ({node.name}): Input tensor x is not 3D (shape {x.shape}).")
            return x # Fallback

        input_feat_dim = x.shape[-1]

        # --- Perform Multiplication ---
        try:
            # --- Case 1: Standard MatMul x @ y ---
            if input_feat_dim == feature_dim:
                # Standard case: (B, S, F) @ (F, F) -> (B, S, F)
                output = torch.matmul(x, y)

            # --- Case 2: Adapted MatMul for Feature Dim 1 Input ---
            elif input_feat_dim == 1:
                # Adapt: Treat as linear expansion using first column of y
                # (B, S, 1) @ (1, F) -> (B, S, F)
                print(f"Adapting {op_name} ({node.name}): Input feat dim is 1. Using first column of y for expansion.")
                # Select first column, keep dim: shape (F, 1)
                y_col = y[:, 0:1]
                # Transpose to (1, F)
                y_col_T = y_col.transpose(-1, -2) # or y_col.T if PyTorch version supports it easily
                output = torch.matmul(x, y_col_T)

            # --- Case 3: Other Input Feature Dimensions (Error) ---
            else:
                # Input feature dimension is neither F nor 1
                print(f"Error in {op_name} ({node.name}): Input feature dim {input_feat_dim} is incompatible with learnable param {y.shape}.")
                return x # Fallback

            return output

        except RuntimeError as e:
            # Catch potential matmul errors even if shapes seem okay initially
            print(f"Error during {op_name} operation ({node.name}): {e}. Shapes: x={x.shape}, y={y.shape}")
            return x # Fallback
        except Exception as e:
            print(f"Unexpected error in {op_name} ({node.name}): {e}")
            return x # Fallback
        
    def inner_prod_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # sum(x * diag(y), dim=-1, keepdim=True) -> (B, S, 1)
        y_vector = self._get_learnable_param_as_vector(node, "inner_prod")
        if y_vector is None: return x
        try:
            elementwise_prod = x * y_vector
            # Sum over the feature dimension
            # Keep dimension to allow broadcasting later, output (B, S, 1)
            inner_product = torch.sum(elementwise_prod, dim=-1, keepdim=True)
            return inner_product
        except RuntimeError as e:
             print(f"Error during inner_prod_core ({node.name}): {e}. Shapes: x={x.shape}, y_vec={y_vector.shape}")
             return x

    # --- Transforms ---
    def fft_mag_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # FFT along sequence dim, return magnitude
        try:
            # Ensure input is float or complex
            if not torch.is_floating_point(x) and not torch.is_complex(x):
                 print(f"Warning: Input to fft_mag_core ({node.name}) is not float/complex ({x.dtype}). Casting to float.")
                 x = x.float()

            # Dim=1 is sequence dimension for (B, S, F)
            fft_result = torch_fft.fft(x, dim=1)
            return torch.abs(fft_result)
        except Exception as e:
            print(f"Error in fft_mag_core ({node.name}): {e}. Input shape: {x.shape}")
            return x

    def ifft_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # Placeholder: If input is real (e.g., magnitude from fft_mag), phase is lost.
        # Just return input for now. A proper IFFT needs complex input.
        # if torch.is_complex(x):
        #     try:
        #         ifft_result = torch_fft.ifft(x, dim=1)
        #         return ifft_result.real # Return real part?
        #     except Exception as e:
        #         print(f"Error in ifft_core ({node.name}): {e}. Input shape: {x.shape}")
        #         return x
        # else:
             # print(f"Warning: Input to ifft_core ({node.name}) is not complex. Returning input directly.")
             return x

    # --- Functional Analysis ---
    def conv1d_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # Input (B, S, F) -> Expected by Conv1d: (B, F, S)
        # Use learnable param y (F, F) as kernel for 1x1 convolution
        if node.learnable_param is None:
            print(f"Error in conv1d_core ({node.name}): learnable_param is None.")
            return x
        y_param = node.learnable_param # Shape (F, F)

        in_channels = self.feature_dim
        out_channels = self.feature_dim
        kernel_size = 1 # Force 1x1 convolution
        padding = 0     # No padding needed for 1x1

        # --- Reshape y_param (F, F) into (F, F, 1) kernel ---
        expected_shape = (out_channels, in_channels, kernel_size)
        kernel_weight = None

        if y_param.shape == (out_channels, in_channels):
            try:
                # Unsqueeze the last dimension to create (F, F, 1)
                kernel_weight = y_param.unsqueeze(-1).contiguous()
            except Exception as e:
                 print(f"Error unsqueezing param ({y_param.shape}) to Conv1d 1x1 kernel {expected_shape} in {node.name}: {e}")
                 return x # Fallback
        else:
             print(f"Error in conv1d_core ({node.name}): Param shape {y_param.shape} not suitable for 1x1 kernel (needs {(out_channels, in_channels)}).")
             return x # Fallback

        # --- Perform 1x1 Convolution ---
        try:
            x_permuted = x.permute(0, 2, 1).contiguous() # (B, F, S)
            output_permuted = nn.functional.conv1d(
                x_permuted,
                weight=kernel_weight, # Use derived (F, F, 1) kernel
                bias=None,
                stride=1,
                padding=padding # Should be 0
            )
            output = output_permuted.permute(0, 2, 1).contiguous() # (B, S, F) - back to original layout
            # Output shape should be same as input for 1x1 conv with padding 0
            if output.shape != x.shape:
                print(f"Warning: Conv1D 1x1 output shape {output.shape} differs from input {x.shape}.")
                # Might need adjustments if stride/padding logic changes later
            return output
        except Exception as e:
             print(f"Error during 1x1 conv1d operation ({node.name}): {e}. Shapes: x={x.shape}, kernel={kernel_weight.shape}")
             return x # Fallback

    # --- Activations & Unary ---
    def tanh_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        return torch.tanh(x)

    def relu_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        return torch.relu(x)

    def sigmoid_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        return torch.sigmoid(x)

    def log_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # Natural log: log(relu(x) + epsilon) for stability
        try:
            safe_x = torch.relu(x) + 1e-8
            return torch.log(safe_x)
        except Exception as e:
            print(f"Error in log_core ({node.name}): {e}. Input shape {x.shape}")
            return x

    # --- Normalization ---
    def layernorm_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        """
        Applies Layer Normalization over the last dimension (features).
        Optionally adds a learnable bias if the input feature dimension matches
        the environment's feature_dim and the node's role is 'bias'.
        """
        op_name = "layernorm_core"
        feature_dim = self.feature_dim # Original F

        if x.ndim < 2: # Need at least Batch and Feature dim
            print(f"Error in {op_name} ({node.name}): Input tensor has too few dimensions ({x.ndim}). Shape: {x.shape}")
            return x

        # Determine the shape over which normalization occurs (last dimension)
        current_feat_dim = x.shape[-1]
        normalized_shape = (current_feat_dim,) # Tuple containing the size of the last dimension

        try:
            # --- Determine Bias Parameter ---
            bias_param = None
            # Check if node is *supposed* to have a bias and has a parameter
            if node.learnable_role == 'bias' and node.learnable_param is not None:
                # Check if the parameter's shape matches the *current* input feature dimension
                if node.learnable_param.shape == normalized_shape:
                    bias_param = node.learnable_param
                    # print(f"Debug: Using LN bias param {bias_param.shape} for input {x.shape}")
                # Check if parameter shape matches original F, but input is different
                elif node.learnable_param.shape == (feature_dim,) and current_feat_dim != feature_dim:
                    print(f"Warning: {op_name} ({node.name}) - Input feature dim {current_feat_dim} differs from bias param dim {feature_dim}. Applying LayerNorm *without* bias.")
                    bias_param = None # Do not use the mismatched bias
                # Check if parameter shape itself is unexpected
                elif node.learnable_param.shape != (feature_dim,):
                    print(f"Warning: {op_name} ({node.name}) - Bias param has unexpected shape {node.learnable_param.shape}. Expected {(feature_dim,)}. Applying LayerNorm *without* bias.")
                    bias_param = None
                # Else: Param shape matches original F and input also has F features (bias_param remains None initially, will be set below if needed)
                # This case is implicitly handled if bias_param is correctly initialized earlier

            # --- Determine Weight (Scale) Parameter (Currently unused but structure is here) ---
            weight_param = None
            # if node.learnable_role == 'scale' and node.learnable_param is not None:
            #     if node.learnable_param.shape == normalized_shape:
            #          weight_param = node.learnable_param
            #     else:
            #          print(f"Warning: {op_name} ({node.name}) - Scale param shape mismatch. Applying LayerNorm *without* scale.")


            # --- Apply LayerNorm ---
            # Provide weight/bias only if they are determined to be compatible
            output = nn.functional.layer_norm(x,
                                            normalized_shape=normalized_shape,
                                            weight=weight_param, # Currently always None
                                            bias=bias_param,     # None if incompatible
                                            eps=1e-5)
            return output

        except Exception as e:
            print(f"Error during {op_name} operation ({node.name}): {e}. Input Shape: {x.shape}, Normalized Shape: {normalized_shape}")
            # traceback.print_exc() # Optional for debugging
            return x # Fallback
    # def layernorm_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
    #     normalized_shape = x.shape[-1:]
    #     try:
    #         bias_param = None
    #         if node.learnable_role == 'bias' and node.learnable_param is not None:
    #              if node.learnable_param.shape == (self.feature_dim,):
    #                   bias_param = node.learnable_param
    #              else: pass # Warning printed elsewhere if role mismatch
    #         # Add scale param handling if defined
    #         weight_param = None
    #         # if node.learnable_role == 'scale' ... weight_param = node.learnable_param ...

    #         return nn.functional.layer_norm(x, normalized_shape, weight=weight_param, bias=bias_param, eps=1e-5)
    #     except Exception as e:
    #          print(f"Error in layernorm_core ({node.name}): {e}. Shape: {x.shape}")
    #          return x

    # --- Order/Analysis (Reductions) ---
    def _reduction_op(self, x: torch.Tensor, node: MathNode, reduction_func: callable, op_name: str) -> torch.Tensor:
        if x is None or x.ndim < 2: # Need at least (B, S) or (S, F) etc.
            print(f"Warning: Input for {op_name} reduction ({node.name}) is not suitable (shape {x.shape if x is not None else 'None'}).")
            return x
        # Reduce along sequence dimension (dim=1 for B,S,F)
        reduction_dim = 1 if x.ndim > 1 else 0 # Reduce along first non-batch dim
        if x.ndim <= reduction_dim:
            print(f"Warning: Input for {op_name} reduction ({node.name}) has insufficient dims ({x.ndim}) for dim {reduction_dim}.")
            return x

        try:
            keep_dim = True # Keep reduced dim for broadcasting compatibility
            if reduction_func == torch.mean:
                 result_reduced = reduction_func(x, dim=reduction_dim, keepdim=keep_dim)
            elif reduction_func in [torch.max, torch.min]:
                 result_reduced, _ = reduction_func(x, dim=reduction_dim, keepdim=keep_dim)
            else: raise ValueError(f"Unsupported reduction function: {reduction_func}")
            return result_reduced
        except Exception as e:
            print(f"Error in {op_name} reduction ({node.name}): {e}. Input shape: {x.shape}")
            return x

    def supremum_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor: # Max
        return self._reduction_op(x, node, torch.max, "supremum")

    def infimum_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor: # Min
        return self._reduction_op(x, node, torch.min, "infimum")

    def mean_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor: # Average
        return self._reduction_op(x, node, torch.mean, "mean")

    # --- Geometry (Unary Elementwise) ---
    def translate_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # x + y_bias
        if node.learnable_role != 'bias' or node.learnable_param is None:
             print(f"Error: Translate node {node.name} requires learnable 'bias' param.")
             return x
        if node.learnable_param.shape != (self.feature_dim,):
             print(f"Error: Translate node {node.name} bias param has wrong shape {node.learnable_param.shape}.")
             return x
        bias = node.learnable_param
        try:
            return x + bias
        except RuntimeError as e:
             print(f"Error during translate_core ({node.name}): {e}. Shapes: x={x.shape}, bias={bias.shape}")
             return x

    def scale_core(self, x: torch.Tensor, node: MathNode) -> torch.Tensor:
        # x * y_scale
        if node.learnable_role != 'scale' or node.learnable_param is None:
             print(f"Error: Scale node {node.name} requires learnable 'scale' param.")
             return x
        if node.learnable_param.shape != (self.feature_dim,):
             print(f"Error: Scale node {node.name} scale param has wrong shape {node.learnable_param.shape}.")
             return x
        scale = node.learnable_param
        try:
            return x * scale
        except RuntimeError as e:
             print(f"Error during scale_core ({node.name}): {e}. Shapes: x={x.shape}, scale={scale.shape}")
             return x

    def close(self):
        """Clean up any resources."""
        print("Closing MathSelfPlayEnv.")
        pass

# --- Training Note (Unchanged) ---
# ... (Keep the note about external training)

# --- Example Usage (Updated for new ops) ---
# if __name__ == '__main__':
#     print("Testing MathSelfPlayEnv with PyTorch backend (Expanded Ops)...")

#     env_config = {
#         'grid_size': 6,        # Slightly larger grid
#         'max_steps': 30,       # More steps
#         'feature_dim': 8,     # Keep feature dim reasonable
#         'batch_size': 16,
#         'sequence_length': 12,
#         'task': 'reverse',
#         'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     }

#     env = MathSelfPlayEnv(**env_config)
#     obs, info = env.reset(seed=123)

#     print(f"Initial Observation (Board Shape): {obs['board'].shape}")
#     env.render()

#     total_reward = 0
#     terminated, truncated = False, False
#     step_count = 0

#     # --- Simple Random Agent Loop ---
#     while not terminated and not truncated:
#         step_count += 1
#         action = env.action_space.sample()

#         print(f"\n--- Step {step_count} (Env Step: {env.steps_taken}) ---")
#         action_op_name = env.operation_types.get(action['operation_id'], 'UnknownOp')
#         print(f"Player {env.current_player} Action: Op={action['operation_id']}:{action_op_name}, Place={action['placement_strategy']}")

#         obs, reward, terminated, truncated, info = env.step(action)

#         print(f"-> Reward: {reward:.4f}, Term: {terminated}, Trunc: {truncated}, Loss: {info.get('last_loss', 'N/A'):.4f}")
#         if info.get('error'): print(f"-> Info/Error: {info['error']}")
#         if info.get('termination_reason'): print(f"-> Term Reason: {info['termination_reason']}")
#         total_reward += reward
#         env.render()
#         # if info.get('error'): # Pause on error
#         #      input("Error encountered. Press Enter to continue...")
#         # elif step_count % 5 == 0: # Pause every 5 steps
#         #      input("Paused. Press Enter...")

#     print("\n--- Episode Finished ---")
#     print(f"Completed in {step_count} agent steps ({env.steps_taken} env steps).")
#     print(f"Total Reward: {total_reward:.4f}")
#     final_node_count = len(env.graph.nodes) if env.graph else 0
#     print(f"Final Nodes: {final_node_count}")

#     # --- Example: Get parameters ---
#     if env.graph and final_node_count > 0:
#         final_params = env.graph.get_parameters()
#         print(f"\nGraph has {len(final_params)} learnable parameter tensors.")

#         # --- Example: Save Structure ---
#         try:
#             graph_struct = env.graph.serialize_graph()
#             print("\nSerialized Graph Structure (JSON):")
#             # print(json.dumps(graph_struct, indent=2)) # Print if needed
#             with open("final_graph_structure_expanded.json", "w") as f:
#                 json.dump(graph_struct, f, indent=2)
#             print("Saved final graph structure to final_graph_structure_expanded.json")
#         except Exception as e:
#             print(f"Error serializing or saving graph: {e}")

#     env.close()