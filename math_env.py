# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from typing import List, Dict, Tuple, Any, Optional
import random
import string # <-- Re-add string import
import math
from scipy import signal
# from scipy.fft import fft, ifft # Use np.fft instead for simplicity if complex handling is tricky
import numpy.fft as np_fft # Use numpy's fft
from scipy import special # For comb (binomial coefficient)
import traceback # For detailed error printing
import json

# --- REMOVE TORCH/TORCHVISION IMPORTS ---
# import torch
# import torch.nn as nn
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# --- END REMOVAL ---


# --- MathNode Class ---
# Use a smaller default feature_dim suitable for sequence tasks
class MathNode:
    """
    Represents a mathematical operation node in the computational graph.
    Each node applies a specific mathematical function to its inputs,
    followed by a learnable affine transformation (W @ result + b).
    """
    def __init__(self, op_id: int, name: str, inputs=None, parameters=None, feature_dim=8, player_id=0): # Smaller default feature_dim
        self.op_id = op_id
        self.name = name
        self.inputs = inputs or []
        self.parameters = parameters or {}
        self.output = None
        self.position = None
        self.output_shape = None
        self.player_id = player_id

        # Initialize learnable parameters with specific dtype
        self.learnable_params = {
            'W': (np.random.randn(feature_dim, feature_dim) * 0.1).astype(np.float32),
            'b': np.zeros(feature_dim, dtype=np.float32)
        }
        self.unique_id = id(self)

    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append(node)

    def set_parameters(self, parameters):
        self.parameters = parameters

    def __repr__(self):
        input_ids = [inp.unique_id for inp in self.inputs]
        return f"MathNode(id={self.unique_id}, name={self.name}, op={self.op_id}, pos={self.position}, player={self.player_id}, inputs={input_ids})"


# --- ComputationalGraph Class ---
class ComputationalGraph:
    """
    Represents a computational graph of mathematical operation nodes.
    Adapted for the self-play environment with a single input and dynamic output.
    """
    def __init__(self):
        self.nodes = []         # All nodes in the graph, ordered by addition
        self.nodes_by_id = {}   # Map unique_id to node
        self.input_node = None  # Single input node (the first node added)
        self.output_node = None # Node designated as output for the current evaluation
        self.grid = {}          # Dictionary mapping (row, col) to node
        self.max_row = -1       # Track max row index used
        self.max_col = -1       # Track max col index used

    def add_node(self, node: MathNode, row: int, col: int):
        """Add a node to the graph at the specified position."""
        if self.get_node_at(row, col) is not None:
            raise ValueError(f"Position ({row},{col}) is already occupied.")

        self.nodes.append(node)
        self.nodes_by_id[node.unique_id] = node
        node.position = (row, col)
        self.grid[(row, col)] = node
        # Update max row/col *after* adding
        self.max_row = max(self.max_row, row)
        self.max_col = max(self.max_col, col)

        # Designate the first node added as the input node
        if len(self.nodes) == 1:
            self.input_node = node
            node.op_id = -1 # Mark as input type
            node.name = f"Input_0_P{node.player_id}" # Include player in name

    def remove_node(self, node: MathNode):
        """Removes a node and cleans up references."""
        if node in self.nodes:
            self.nodes.remove(node)
        if node.unique_id in self.nodes_by_id:
            del self.nodes_by_id[node.unique_id]
        if node.position in self.grid:
            del self.grid[node.position]
        if node == self.input_node:
            self.input_node = None # Should only happen if reverting first move
        if node == self.output_node:
            self.output_node = None # Reset output if removed

        # Remove connections pointing TO this node
        for other_node in self.nodes:
            if node in other_node.inputs:
                other_node.inputs.remove(node)

        # Recalculate max_row/max_col (simple way)
        self.max_row = -1
        self.max_col = -1
        for r, c in self.grid.keys():
            self.max_row = max(self.max_row, r)
            self.max_col = max(self.max_col, c)


    def get_node_at(self, row: int, col: int) -> Optional[MathNode]:
        """Get the node at a specific position, or None."""
        return self.grid.get((row, col))

    def get_node_by_id(self, unique_id: int) -> Optional[MathNode]:
        return self.nodes_by_id.get(unique_id)

    def connect_nodes(self, source_node_id: int, target_node_id: int) -> bool:
        """Connect nodes by their unique IDs."""
        source = self.get_node_by_id(source_node_id)
        target = self.get_node_by_id(target_node_id)

        if source and target:
            target.add_input(source)
            return True
        print(f"Warning: Could not connect {source_node_id} to {target_node_id}. Source or target not found.")
        return False

    def set_output_node(self, node: MathNode):
        """Designate a specific node as the output for evaluation."""
        if node in self.nodes:
            self.output_node = node
        else:
            print(f"Warning: Cannot set output node {node}. Node not found in graph.")
            self.output_node = None

    def is_valid_dag(self) -> bool:
        """Check if the graph is a valid DAG."""
        visited = set()
        recursion_stack = set()

        def is_cyclic_util(node_id):
            node = self.get_node_by_id(node_id)
            if not node: return False # Should not happen

            visited.add(node_id)
            recursion_stack.add(node_id)

            for neighbor in node.inputs:
                neighbor_id = neighbor.unique_id
                if neighbor_id not in visited:
                    if is_cyclic_util(neighbor_id):
                        return True
                elif neighbor_id in recursion_stack:
                    return True # Cycle detected

            recursion_stack.remove(node_id)
            return False

        # Check cycles starting from all nodes
        # Optimization: Only need to check from nodes with no outgoing edges (potential cycle ends)
        # or nodes that were recently added/modified. For simplicity, checking all is safer.
        node_ids = list(self.nodes_by_id.keys())
        for node_id in node_ids:
            if node_id not in visited:
                if is_cyclic_util(node_id):
                    return False
        return True

    def topological_sort(self) -> List[MathNode]:
        """Return nodes in topological order."""
        visited = set()
        topo_order = []
        visiting = set() # To detect cycles during sort

        def visit(node_id):
            node = self.get_node_by_id(node_id)
            if not node: return # Node might have been removed
            if node_id in visiting:
                raise RuntimeError("Cycle detected during topological sort") # Should be caught by is_valid_dag earlier
            if node_id in visited: return

            visiting.add(node_id)
            visited.add(node_id)

            # Visit dependencies first
            for inp in node.inputs:
                visit(inp.unique_id)

            visiting.remove(node_id)
            topo_order.append(node)

        # Ensure all nodes are considered
        node_ids = list(self.nodes_by_id.keys()) # Use a copy of keys
        for node_id in node_ids:
             if node_id not in visited:
                 visit(node_id)

        return topo_order

    # --- REMOVED input_embedding from forward_pass signature ---
    def forward_pass(self, input_tensor: np.ndarray, operations: Dict[int, Dict]) -> Optional[np.ndarray]:
        """
        Perform forward pass. Assumes a single input tensor and a single designated output node.
        Input tensor is now assumed to be (batch, seq_len, feature_dim).
        """
        if not self.input_node:
            print("Error: Input node not set in graph.")
            return None
        if not self.output_node:
             # If no output node is explicitly set, maybe default to the last added node?
             # Or require it to be set before calling forward_pass.
             # Current behavior: return None if output_node is not set.
             print("Error: Output node not set in graph.")
             return None

        # Reset outputs
        for node in self.nodes:
            node.output = None
            node.output_shape = None

        # --- Assign Raw Input Tensor ---
        try:
            # input_tensor shape: (batch, seq_len, feature_dim)
            if not isinstance(input_tensor, np.ndarray):
                 raise TypeError(f"Input tensor must be a numpy array, got {type(input_tensor)}")
            if input_tensor.ndim != 3:
                 raise ValueError(f"Input tensor must have 3 dimensions (batch, seq, feat), got {input_tensor.ndim}")

            self.input_node.output = input_tensor.copy().astype(np.float32) # Ensure float32
            self.input_node.output_shape = input_tensor.shape
        except Exception as e:
            print(f"Error assigning input tensor: {e}")
            traceback.print_exc()
            return None
        # --- End Assign Input ---


        sorted_nodes = self.topological_sort()
        final_output = None # This will be the output of the designated self.output_node

        for node in sorted_nodes:
            if node == self.input_node:
                continue # Already assigned

            # Collect inputs (ensure they are available)
            node_inputs = []
            inputs_ready = True
            if not node.inputs:
                 # Nodes other than input must have inputs to be evaluated
                 # (Unless it's a constant node, which we don't have explicitly)
                 if node != self.input_node:
                      # print(f"Warning: Node {node.name} has no inputs.") # Optional warning
                      inputs_ready = False # Cannot evaluate without inputs
            else:
                 for inp_node in node.inputs:
                      if inp_node.output is None:
                           # This can happen if an input node failed to compute its output
                           print(f"Error: Input {inp_node.name} ({inp_node.unique_id}) for node {node.name} ({node.unique_id}) is None.")
                           inputs_ready = False
                           break
                      node_inputs.append(inp_node.output)

            if not inputs_ready:
                 # print(f"Skipping node {node.name} due to missing inputs.") # Debugging
                 node.output = None # Mark as failed
                 continue

            # Apply operation (using the _apply_operation helper from the env)
            if node.op_id in operations:
                try:
                    op_info = operations[node.op_id]
                    apply_func = op_info['apply']
                    core_func = op_info['core']
                    is_elementwise = op_info.get('elementwise', True)

                    # Ensure learnable params are float32
                    node.learnable_params['W'] = node.learnable_params['W'].astype(np.float32)
                    node.learnable_params['b'] = node.learnable_params['b'].astype(np.float32)

                    op_params = {
                        'inputs': node_inputs, # List of np.ndarrays
                        'learnable_params': node.learnable_params,
                        **node.parameters
                    }

                    node.output = apply_func(node_inputs, core_func, op_params, is_elementwise)

                    if node.output is not None:
                        node.output_shape = node.output.shape
                        # Ensure output is float32
                        if node.output.dtype != np.float32:
                            # print(f"Warning: Node {node.name} output dtype was {node.output.dtype}, converting to float32.")
                            node.output = node.output.astype(np.float32)
                    # else:
                    #      print(f"Warning: Node {node.name} op {node.op_id} produced None output.") # Debugging

                except Exception as e:
                    print(f"Error evaluating node {node.name} ({node.unique_id}) op {node.op_id}: {str(e)}")
                    traceback.print_exc()
                    node.output = None # Ensure output is None on error
            else:
                 print(f"Warning: Operation ID {node.op_id} not found in operations implementation for node {node.name}.")
                 node.output = None


            # Check if this is the designated output node
            if node == self.output_node:
                final_output = node.output
                # print(f"Output node {node.name} evaluated. Output shape: {node.output_shape}") # Debugging

        # Return the raw output of the final graph node
        if final_output is None and self.output_node is not None:
             # Check if the output node itself failed
             if self.output_node.output is None:
                  print(f"Warning: Final output is None because the designated output node ({self.output_node.name}) failed evaluation.")
             else:
                  # This case should ideally not happen if topological sort is correct
                  print(f"Warning: Final output is None, but output node ({self.output_node.name}) was designated and evaluated (output shape: {self.output_node.output_shape}). Check logic.")
        # elif final_output is not None:
        #      print(f"Forward pass completed. Final output shape: {final_output.shape}") # Debugging

        return final_output

    # --- Graph Serialization ---
    def serialize_graph(self) -> List[Dict]:
        """
        Serializes the current graph structure into a list of dictionaries.
        Each dictionary represents a node and its connections.
        """
        serialized_nodes = []
        for node in self.nodes:
            node_data = {
                "unique_id": node.unique_id,
                "op_id": node.op_id, # Use the renumbered ID
                "name": node.name, # Optional: include name for readability
                "position": node.position, # Tuple (row, col)
                "player_id": node.player_id,
                "input_ids": [inp.unique_id for inp in node.inputs] # List of IDs of input nodes
            }
            serialized_nodes.append(node_data)
        return serialized_nodes

# --- MathSelfPlayEnv Class ---
class MathSelfPlayEnv(gym.Env):
    """
    Gym environment for self-play graph construction for sequence-to-sequence tasks.
    Players take turns placing learnable math nodes on a grid.
    Reward is based on sequence loss (MSE) and expansion penalty.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- Reward Weights ---
    ACCURACY_PENALTY_WEIGHT = 2.0 # Adjusted weight for MSE loss
    EXPANSION_PENALTY = 0.01      # Penalty for increasing grid dimensions
    INVALID_MOVE_PENALTY = -1.0   # Penalty for placing on occupied cell, off-grid, or creating cycle
    EVAL_FAILURE_PENALTY = -0.7   # Penalty if graph forward pass fails or returns None
    SHAPE_MISMATCH_PENALTY = -0.5 # Penalty if output shape doesn't match target
    NAN_INF_LOSS_PENALTY = -1.0   # Penalty if loss calculation results in NaN/Inf
    LOSS_CALC_ERROR_PENALTY = -0.8 # Penalty for other errors during loss calculation
    UNEXPECTED_STEP_ERROR_PENALTY = -2.0 # Penalty for unexpected errors in step()

    # --- MODIFIED __init__ SIGNATURE ---
    def __init__(self,
                 grid_size=10,       # Default max grid size
                 max_steps=50,       # Default max steps per episode
                 feature_dim=8,      # Default feature dim for sequences
                 batch_size=64,
                 sequence_length=15, # Added sequence_length back
                 task='addition'):   # Added task parameter ('addition' or 'reverse')
        super().__init__()

        # --- Use the provided or default values ---
        self.grid_size = grid_size
        self.max_steps = max_steps
        # --- End Use ---

        self.max_nodes = self.grid_size * self.grid_size
        self.feature_dim = feature_dim
        self.batch_size = batch_size
        self.sequence_length = sequence_length # Store sequence length
        self.task = task # Store the task type

        # --- Add back char_to_point for sequence generation ---
        self.char_to_point: Optional[Dict[str, np.ndarray]] = None
        self.PAD_CHAR = ' ' # Define padding character consistently
        self.UNK_CHAR = '<UNK>' # Define unknown character consistently
        # --- End Add back ---

        # --- Define Operations ---
        # ... (Operation definition code remains the same) ...
        _original_ops = {
            0: "Addition", 1: "Subtraction", 2: "Multiplication", 3: "Division",
            4: "Exponentiation", 5: "Root Extraction", 6: "Derivative", 7: "Integral",
            8: "Fourier Transform", 9: "Inverse Fourier Transform", 10: "Laplace Transform",
            11: "Inverse Laplace Transform", 12: "Z-Transform", 13: "Inverse Z-Transform",
            14: "Convolution", 15: "Set Union", 16: "Set Intersection", 17: "Set Complement",
            18: "Cartesian Product", 19: "Logical AND", 20: "Logical OR", 21: "Logical NOT",
            22: "Quantifiers", 23: "Function Composition", 24: "Vector Addition",
            25: "Scalar Multiplication", 26: "Inner Product", 27: "Matrix Multiplication",
            28: "GCD", 29: "Modulo", 30: "Factorial", 31: "Binomial Coefficient",
            32: "Closure", 33: "Supremum", 34: "Infimum", 35: "Measure",
            36: "Distance", 37: "Rotation", 38: "Reflection", 39: "Translation",
            40: "Entropy"
        }
        _ids_to_remove = {10, 11, 12, 13, 18, 22, 23, 32} # Remove less common/complex ops
        self.operation_types = {}
        _new_id_counter = 0
        self._original_to_new_id_map = {}
        for old_id, name in _original_ops.items():
            if old_id not in _ids_to_remove:
                self.operation_types[_new_id_counter] = name
                self._original_to_new_id_map[old_id] = _new_id_counter
                _new_id_counter += 1
        self.num_operations = len(self.operation_types)

        self.operations_impl = {}
        for op_id, name in self.operation_types.items():
             method_name = name.lower().replace(" ", "_").replace("-","_")
             core_func = getattr(self, method_name + "_core", self._placeholder_op_core)
             # Determine if operation works elementwise or on the whole sequence
             is_elementwise = not (name in ["Derivative", "Integral", "Fourier Transform",
                                            "Inverse Fourier Transform", "Convolution", "Entropy",
                                            "Measure", "Supremum", "Infimum"])
             self.operations_impl[op_id] = {
                 'core': core_func, 'apply': self._apply_operation,
                 'elementwise': is_elementwise, 'name': name
             }
        # Alias Vector Addition to use the same core as Addition
        vector_add_new_id = self._original_to_new_id_map.get(24)
        add_new_id = self._original_to_new_id_map.get(0)
        if vector_add_new_id is not None and add_new_id is not None:
             self.operations_impl[vector_add_new_id]['core'] = self.addition_core


        # --- Action Space ---
        self.num_placement_strategies = 5 # 0: Below Input, 1: Up, 2: Right, 3: Down, 4: Left (relative to pointer)
        self.action_space = spaces.Dict({
            'operation_id': spaces.Discrete(self.num_operations),
            'placement_strategy': spaces.Discrete(self.num_placement_strategies)
        })

        # --- Observation Space ---
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

        # Internal state variables
        self.graph: Optional[ComputationalGraph] = None
        self.current_player: int = 1
        self.pointer_location: Optional[Tuple[int, int]] = None
        self.last_loss: float = float('inf') # Use infinity for initial loss
        self.steps_taken: int = 0
        self.current_inputs: Optional[np.ndarray] = None # Shape: (batch, seq_len, feature_dim)
        self.target_outputs: Optional[np.ndarray] = None # Shape: (batch, seq_len, feature_dim)

        # Initialize char_to_point mapping based on the task
        self._initialize_char_to_point()

    def _initialize_char_to_point(self):
        """Initializes the character-to-vector mapping based on the task."""
        self.char_to_point = {}
        rng = self.np_random if hasattr(self, 'np_random') else np.random.default_rng()

        if self.task == 'addition':
            chars = string.digits + '+'
        elif self.task == 'reverse':
            chars = string.ascii_lowercase + string.digits + " " # Use space for this task too
        else:
            raise ValueError(f"Unknown task: {self.task}")

        all_chars_for_map = set(list(chars))
        all_chars_for_map.add(self.PAD_CHAR)
        all_chars_for_map.add(self.UNK_CHAR)

        for char in all_chars_for_map:
            if char == self.PAD_CHAR or char == self.UNK_CHAR:
                point = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                point = rng.uniform(-1, 1, size=self.feature_dim).astype(np.float32)
            self.char_to_point[char] = point
        print(f"Initialized char_to_point for task '{self.task}' with {len(self.char_to_point)} characters.")


    def _generate_addition_data(self):
        """Generate sequence data for addition task."""
        if self.char_to_point is None:
            self._initialize_char_to_point() # Should already be called by __init__

        digits = string.digits  # '0' to '9'
        # Adjust max_num calculation to prevent overly long input strings before padding
        max_digits_per_num = (self.sequence_length - 1) // 2 # -1 for '+' sign
        max_num = 10 ** max_digits_per_num - 1
        if max_num <= 0:
             raise ValueError(f"Sequence length {self.sequence_length} too short for addition task.")

        # Initialize input and target tensors
        self.current_inputs = np.zeros((self.batch_size, self.sequence_length, self.feature_dim), dtype=np.float32)
        self.target_outputs = np.zeros_like(self.current_inputs)

        # Get the vector for unknown characters once
        unk_vector = self.char_to_point.get(self.UNK_CHAR, np.zeros(self.feature_dim, dtype=np.float32))

        for i in range(self.batch_size):
            try:
                # Generate two random numbers
                num1 = self.np_random.integers(0, max_num + 1) # Use max_num+1 for inclusive range
                num2 = self.np_random.integers(0, max_num + 1)

                # Convert numbers to strings
                num1_str = str(num1)
                num2_str = str(num2)

                # Create input sequence (num1 + num2)
                input_seq_unpadded = num1_str + '+' + num2_str
                if len(input_seq_unpadded) > self.sequence_length:
                    # This should not happen with the corrected max_num, but as a safeguard:
                    print(f"Warning: Generated input sequence '{input_seq_unpadded}' longer than sequence length {self.sequence_length}. Truncating.")
                    input_seq_unpadded = input_seq_unpadded[:self.sequence_length]

                # Pad with the single padding character
                input_seq = input_seq_unpadded.ljust(self.sequence_length, self.PAD_CHAR)

                # Create target sequence (sum)
                target_sum = num1 + num2
                target_seq_unpadded = str(target_sum)
                if len(target_seq_unpadded) > self.sequence_length:
                     # The sum might exceed the sequence length
                     print(f"Warning: Generated target sequence '{target_seq_unpadded}' longer than sequence length {self.sequence_length}. Truncating.")
                     target_seq_unpadded = target_seq_unpadded[:self.sequence_length]

                # Pad with the single padding character
                target_seq = target_seq_unpadded.ljust(self.sequence_length, self.PAD_CHAR)

                # Populate tensors using char_to_point mapping
                for j in range(self.sequence_length):
                    input_char = input_seq[j]
                    target_char = target_seq[j]
                    # Use .get() with fallback to UNK vector
                    self.current_inputs[i, j] = self.char_to_point.get(input_char, unk_vector)
                    self.target_outputs[i, j] = self.char_to_point.get(target_char, unk_vector)

            except Exception as e:
                 print(f"Error generating data for batch item {i}: {e}")
                 traceback.print_exc()
                 # Fill with padding/zeros to avoid downstream errors
                 self.current_inputs[i, :, :] = self.char_to_point.get(self.PAD_CHAR, unk_vector)
                 self.target_outputs[i, :, :] = self.char_to_point.get(self.PAD_CHAR, unk_vector)


    def _generate_char_sequence_data(self):
        """Generate character sequence data mapped to feature_dim vectors (reverse task)."""
        if self.char_to_point is None:
            self._initialize_char_to_point() # Should already be called by __init__

        # Use the characters defined during initialization for this task
        chars_for_task = string.ascii_lowercase + string.digits + " "
        char_list = list(chars_for_task) # Convert to list for choice

        # Initialize input and target tensors
        self.current_inputs = np.zeros((self.batch_size, self.sequence_length, self.feature_dim), dtype=np.float32)
        self.target_outputs = np.zeros_like(self.current_inputs)

        # Get the vector for unknown characters once
        unk_vector = self.char_to_point.get(self.UNK_CHAR, np.zeros(self.feature_dim, dtype=np.float32))
        pad_vector = self.char_to_point.get(self.PAD_CHAR, unk_vector) # Use UNK if PAD somehow missing

        rng = self.np_random if hasattr(self, 'np_random') else np.random.default_rng()

        for i in range(self.batch_size):
            try:
                # Generate random phrases for the batch
                length = rng.integers(self.sequence_length // 2, self.sequence_length + 1)
                phrase = ''.join(rng.choice(char_list, size=length))

                # Simple target: reversed sequence
                target_phrase = phrase[::-1]

                # Populate tensors
                for j in range(self.sequence_length):
                    # Input sequence (pad with PAD_CHAR)
                    char_in = phrase[j] if j < len(phrase) else self.PAD_CHAR
                    self.current_inputs[i, j] = self.char_to_point.get(char_in, unk_vector)
                    # Target sequence (pad with PAD_CHAR)
                    char_out = target_phrase[j] if j < len(target_phrase) else self.PAD_CHAR
                    self.target_outputs[i, j] = self.char_to_point.get(char_out, unk_vector)

            except Exception as e:
                 print(f"Error generating data for batch item {i}: {e}")
                 traceback.print_exc()
                 # Fill with padding/zeros
                 self.current_inputs[i, :, :] = pad_vector
                 self.target_outputs[i, :, :] = pad_vector


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.graph = ComputationalGraph()
        self.current_player = 1
        self.pointer_location = None
        self.last_loss = 10.0 # Initial high loss guess (MSE can be large)
        self.steps_taken = 0

        # --- Generate sequence data based on the task ---
        if self.task == 'addition':
            self._generate_addition_data()
        elif self.task == 'reverse':
            self._generate_char_sequence_data()
        else:
            raise ValueError(f"Unknown task type '{self.task}' for data generation.")

        # Ensure data was generated
        if self.current_inputs is None or self.target_outputs is None:
             raise RuntimeError(f"Data generation failed for task '{self.task}'. Check sequence_length and batch_size.")

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: Dict[str, int]):
        """Take a turn in the game by placing a node."""
        if self.graph is None:
            raise RuntimeError("Environment needs to be reset before stepping.")
        if self.current_inputs is None or self.target_outputs is None:
             raise RuntimeError("Environment data (current_inputs/target_outputs) not initialized. Call reset first.")

        operation_id = action['operation_id']
        placement_strategy = action['placement_strategy']

        terminated = False
        truncated = False # Initialize truncated flag
        reward = 0.0
        info = {'error': '', 'termination_reason': None}
        new_node = None

        previous_pointer_location = self.pointer_location
        prev_max_row = self.graph.max_row
        prev_max_col = self.graph.max_col

        try:
            # --- 1. Determine Target Position ---
            target_row, target_col = -1, -1
            is_first_move = (len(self.graph.nodes) == 0)

            if is_first_move:
                # First move always places the input node at (0, 0)
                # The action dict contains the *first operation* node's details
                target_row, target_col = 0, 0
                # Create the actual input node first
                input_node = MathNode(
                    op_id=-1, name=f"Input_0_P{self.current_player}",
                    feature_dim=self.feature_dim, player_id=self.current_player
                )
                self.graph.add_node(input_node, target_row, target_col)
                self.pointer_location = (target_row, target_col) # Pointer starts at input node
                previous_pointer_location = self.pointer_location # Update for connection logic

                # Now determine position for the *first operation* node based on strategy
                # Strategy 0 is invalid for the *first operation* node placement
                if placement_strategy == 0:
                     raise ValueError("Placement strategy 0 (relative to input) is invalid for the very first operation node.")
                elif placement_strategy == 1: target_row, target_col = -1, 0 # Invalid (Up from 0,0)
                elif placement_strategy == 2: target_row, target_col = 0, 1 # Right of input
                elif placement_strategy == 3: target_row, target_col = 1, 0 # Down from input
                elif placement_strategy == 4: target_row, target_col = 0, -1 # Invalid (Left from 0,0)
                else: raise ValueError(f"Unknown placement strategy: {placement_strategy}")

            else: # Not the first move (input node already exists)
                if self.pointer_location is None:
                     # This case should ideally not happen after the first move logic above
                     raise RuntimeError("Pointer location is None after the first move.")
                else:
                     pointer_row, pointer_col = self.pointer_location

                # Placement strategies (relative to pointer)
                if placement_strategy == 1: target_row, target_col = pointer_row - 1, pointer_col # Up
                elif placement_strategy == 2: target_row, target_col = pointer_row, pointer_col + 1 # Right
                elif placement_strategy == 3: target_row, target_col = pointer_row + 1, pointer_col # Down
                elif placement_strategy == 4: target_row, target_col = pointer_row, pointer_col - 1 # Left
                elif placement_strategy == 0: # Strategy 0: Place relative to input node
                     if self.graph.input_node and self.graph.input_node.position:
                          input_r, input_c = self.graph.input_node.position
                          # Try placing below input node first
                          target_row, target_col = input_r + 1, input_c
                          # If below is occupied or off-grid, try right
                          if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size) or self.graph.get_node_at(target_row, target_col):
                               target_row, target_col = input_r, input_c + 1
                          # If right is also occupied or off-grid, consider it invalid for simplicity
                          if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size) or self.graph.get_node_at(target_row, target_col):
                               raise ValueError("Placement strategy 0 failed: Positions relative to input node are occupied or off-grid.")
                     else:
                          raise ValueError("Placement strategy 0 invalid: Input node position unknown.")
                else: raise ValueError(f"Unknown placement strategy: {placement_strategy}")

            # --- 2. Validate Position ---
            if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size):
                raise ValueError(f"Invalid move: Position ({target_row},{target_col}) is off-grid ({self.grid_size}x{self.grid_size}).")
            occupied_node = self.graph.get_node_at(target_row, target_col)
            if occupied_node is not None:
                raise ValueError(f"Invalid move: Position ({target_row},{target_col}) is occupied by {occupied_node.name}.")

            # --- 3. Create and Add Operation Node ---
            if not (0 <= operation_id < self.num_operations):
                 raise ValueError(f"Invalid operation_id: {operation_id} (should be 0-{self.num_operations-1})")

            op_name = self.operations_impl[operation_id]['name']
            node_name = f"{op_name}_{len(self.graph.nodes)}_P{self.current_player}"
            new_node = MathNode(
                op_id=operation_id, name=node_name,
                feature_dim=self.feature_dim, player_id=self.current_player
            )
            self.graph.add_node(new_node, target_row, target_col)

            # --- 4. Connect Node ---
            # Connect from the node at the previous pointer location.
            # previous_pointer_location was set correctly even for the first move.
            source_node = None
            if previous_pointer_location:
                 source_node = self.graph.get_node_at(*previous_pointer_location)

            if source_node:
                 self.graph.connect_nodes(source_node.unique_id, new_node.unique_id)
            else:
                 # This should not happen if previous_pointer_location is always valid after first move
                 print(f"Critical Warning: Could not find source node at {previous_pointer_location} to connect to {new_node.name}")
                 # Attempt to connect from input node as a last resort if it exists and isn't the new node
                 if self.graph.input_node and self.graph.input_node != new_node:
                      print(f"Attempting fallback connection from input node.")
                      self.graph.connect_nodes(self.graph.input_node.unique_id, new_node.unique_id)
                 else:
                      # If connection fails completely, the node is isolated. This might be okay
                      # depending on the desired graph structures, but likely indicates an issue.
                      print(f"Warning: Node {new_node.name} added but could not be connected.")


            # --- 5. Update Pointer ---
            self.pointer_location = (target_row, target_col)

            # --- 6. Check DAG ---
            if not self.graph.is_valid_dag():
                # Revert changes if cycle created
                self.graph.remove_node(new_node) # remove_node handles grid/list/id cleanup
                self.pointer_location = previous_pointer_location # Restore pointer
                new_node = None # Ensure new_node is None so it's not used further
                raise ValueError("Invalid move: Created a cycle.")

            # --- 7. Evaluate Graph & Calculate Reward ---
            self.graph.set_output_node(new_node) # Evaluate based on the newly added node
            current_loss = float('inf')
            eval_output_np = None # Initialize to None

            # Ensure graph has input and output nodes before evaluation
            if self.graph.input_node and self.graph.output_node:
                 eval_output_np = self.graph.forward_pass(self.current_inputs, self.operations_impl)
            else:
                 info['error'] = "Graph evaluation skipped: Missing input or output node."
                 # Assign high loss if evaluation skipped
                 current_loss = (self.last_loss + 20.0) if np.isfinite(self.last_loss) else 100.0
                 reward = self.EVAL_FAILURE_PENALTY


            # --- Reward Calculation Logic (Sequence MSE) ---
            if eval_output_np is not None:
                # <<< --- START MODIFICATION --- >>>
                # Clip the output values to prevent overflow during loss calculation
                CLIP_VALUE = 1e6 # Define a reasonable clipping value
                eval_output_np = np.clip(eval_output_np, -CLIP_VALUE, CLIP_VALUE)
                # <<< --- END MODIFICATION --- >>>

                # Expected output shape: (batch, seq_len, feature_dim)
                if eval_output_np.shape == self.target_outputs.shape:
                    try:
                        # Calculate MSE loss
                        diff = eval_output_np - self.target_outputs
                        loss = np.mean(diff * diff) # Use mean squared error

                        if np.isnan(loss) or np.isinf(loss):
                            current_loss = 100.0 # Assign large finite loss for bad numerics
                            reward = self.NAN_INF_LOSS_PENALTY
                            info['error'] = "Evaluation resulted in NaN/Inf loss."
                        else:
                            current_loss = loss
                            # Reward = Improvement - Accuracy Penalty - Expansion Penalty
                            finite_last_loss = self.last_loss if np.isfinite(self.last_loss) else 100.0 # Use a large number if last loss was inf
                            improvement = finite_last_loss - current_loss
                            accuracy_penalty = self.ACCURACY_PENALTY_WEIGHT * current_loss
                            reward = improvement - accuracy_penalty

                    except Exception as loss_calc_e:
                         current_loss = 100.0
                         reward = self.LOSS_CALC_ERROR_PENALTY
                         info['error'] = f"Error during loss calculation: {loss_calc_e}"
                         traceback.print_exc()

                else:
                    # Shape mismatch penalty
                    current_loss = (self.last_loss + 10.0) if np.isfinite(self.last_loss) else 100.0
                    reward = self.SHAPE_MISMATCH_PENALTY
                    info['error'] = f"Output shape mismatch: Expected {self.target_outputs.shape}, Got {eval_output_np.shape}"
            # Check if eval failed AND no specific error was set yet
            elif 'error' not in info or not info['error']:
                # Forward pass failed or produced None output without specific reason caught above
                current_loss = (self.last_loss + 20.0) if np.isfinite(self.last_loss) else 100.0
                reward = self.EVAL_FAILURE_PENALTY
                info['error'] = "Graph evaluation failed or produced None output."

            # --- Apply Expansion Penalty ---
            expanded_grid = (self.graph.max_row > prev_max_row) or (self.graph.max_col > prev_max_col)
            if expanded_grid:
                reward -= self.EXPANSION_PENALTY
                info['grid_expanded'] = True

            self.last_loss = current_loss
            # --- End Reward Calculation ---

        except ValueError as e: # Catch specific ValueErrors from validation/placement/cycle
            reward = self.INVALID_MOVE_PENALTY
            info['error'] = str(e)
            # Restore pointer if move failed due to invalid placement/cycle etc.
            self.pointer_location = previous_pointer_location
            # Ensure graph state is consistent (new_node should not be in graph if error occurred before adding/connecting)
            if new_node and new_node in self.graph.nodes:
                 self.graph.remove_node(new_node) # Clean up if node was added before error

        except Exception as e: # Catch unexpected errors during step
            reward = self.UNEXPECTED_STEP_ERROR_PENALTY
            info['error'] = f"Unexpected error in step: {str(e)}"
            traceback.print_exc()
            terminated = True # End episode on unexpected error
            info['termination_reason'] = 'unexpected_error'
            # Attempt to restore state
            self.pointer_location = previous_pointer_location
            if new_node and new_node in self.graph.nodes:
                 self.graph.remove_node(new_node)

        # --- 8. Update Step Counter and Switch Player ---
        self.steps_taken += 1
        self.current_player = 3 - self.current_player # Switch between 1 and 2

        # --- 9. Check Termination Conditions ---
        if not terminated: # Only check these if not already terminated by error
            if self.steps_taken >= self.max_steps:
                terminated = True
                info['termination_reason'] = 'max_steps_reached'
            elif len(self.graph.nodes) >= self.max_nodes:
                 terminated = True
                 info['termination_reason'] = 'max_nodes_reached'
            # Add termination based on loss? (e.g., if loss is extremely low)
            # elif np.isfinite(self.last_loss) and self.last_loss < 1e-5:
            #      terminated = True
            #      info['termination_reason'] = 'loss_threshold_reached'


        # --- 10. Prepare return values ---
        observation = self._get_observation()
        # Ensure reward is finite
        if not np.isfinite(reward):
             print(f"Warning: Non-finite reward calculated ({reward}). Clamping to {self.UNEXPECTED_STEP_ERROR_PENALTY}.")
             reward = self.UNEXPECTED_STEP_ERROR_PENALTY

        # Use truncated flag if termination is due to time limit (max_steps) rather than task completion
        truncated = (info.get('termination_reason') == 'max_steps_reached')
        if truncated:
             terminated = False # Gymnasium standard: truncated=True means terminated=False

        info['last_loss'] = self.last_loss if np.isfinite(self.last_loss) else 100.0 # Report finite loss

        return observation, reward, terminated, truncated, info

    # --- _get_observation, _get_info, render remain the same ---
    def _get_observation(self):
        """Construct the observation dictionary."""
        board_shape = self.observation_space['board'].shape
        board = np.zeros(board_shape, dtype=np.float32)

        if self.graph is None: # Should not happen after reset
             print("Warning: _get_observation called with self.graph=None")
             return {
                'board': board,
                'current_player': self.current_player,
                'steps_taken': self.steps_taken
             }

        # Channel indices based on updated observation space definition
        op_channel_offset = 0 # Channels 0 to num_operations-1
        input_channel_idx = self.num_operations # Channel for input node marker
        player1_channel_idx = self.num_operations + 1 # Channel for player 1 nodes
        player2_channel_idx = self.num_operations + 2 # Channel for player 2 nodes
        pointer_channel_idx = self.num_operations + 3 # Channel for pointer location

        for node in self.graph.nodes:
            if node.position is None:
                 print(f"Warning: Node {node.name} has no position in _get_observation.")
                 continue # Should not happen for added nodes
            r, c = node.position
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                # Mark operation type or input node
                if node.op_id == -1: # Input node
                    board[r, c, input_channel_idx] = 1.0
                elif 0 <= node.op_id < self.num_operations: # Check against new range
                    board[r, c, op_channel_offset + node.op_id] = 1.0
                # else: # Should not happen if op_id validation is correct
                #     print(f"Warning: Node {node.name} has invalid op_id {node.op_id} in _get_observation.")


                # Mark which player placed the node
                if node.player_id == 1:
                    board[r, c, player1_channel_idx] = 1.0
                elif node.player_id == 2:
                    board[r, c, player2_channel_idx] = 1.0

        # Mark pointer location
        if self.pointer_location:
            pr, pc = self.pointer_location
            if 0 <= pr < self.grid_size and 0 <= pc < self.grid_size:
                board[pr, pc, pointer_channel_idx] = 1.0
            # else: # Pointer should always be within grid if placement validation works
            #     print(f"Warning: Pointer location {self.pointer_location} is outside grid bounds.")


        return {
            'board': board,
            'current_player': self.current_player,
            'steps_taken': self.steps_taken
        }

    def _get_info(self):
        """Return auxiliary information about the environment state."""
        finite_loss = self.last_loss if np.isfinite(self.last_loss) else 100.0
        return {
            'last_loss': finite_loss,
            'nodes_count': len(self.graph.nodes) if self.graph else 0,
            'pointer': self.pointer_location,
            'max_row': self.graph.max_row if self.graph else -1,
            'max_col': self.graph.max_col if self.graph else -1,
        }

    def render(self, mode='human'):
        """Render the current state of the environment grid."""
        if mode != 'human' or self.graph is None:
            return

        print("-" * (self.grid_size * 7)) # Adjusted width
        print(f"Step: {self.steps_taken}/{self.max_steps}, Player: {self.current_player}'s Turn")
        loss_val = self.last_loss if np.isfinite(self.last_loss) else float('inf')
        print(f"Last Eval Loss: {loss_val:.4f}")
        print(f"Pointer Location: {self.pointer_location}")
        print(f"Nodes: {len(self.graph.nodes)}/{self.max_nodes}")

        grid_repr = [['.' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                node = self.graph.get_node_at(r, c)
                if node:
                    op_char = str(node.op_id) if node.op_id != -1 else 'In' # Display new ID
                    player_mark = f"P{node.player_id}"
                    cell_str = f"{op_char}({player_mark})"
                    if (r, c) == self.pointer_location:
                        cell_str += "*" # Mark pointer
                    grid_repr[r][c] = cell_str

        print("\nBoard:")
        header = "  " + " ".join(f"{i:^6}" for i in range(self.grid_size))
        print(header)
        print("  " + "-" * (self.grid_size * 7 - 1))
        for r in range(self.grid_size):
            print(f"{r}|" + "|".join(f"{cell:^6}" for cell in grid_repr[r]) + "|")
        print("  " + "-" * (self.grid_size * 7 - 1))

        print("-" * (self.grid_size * 7))


    # --- _apply_operation and _core functions remain the same ---
    # Note: They now operate on self.feature_dim (e.g., 8)
    def _apply_operation(self, inputs: List[np.ndarray], operation_func: callable, params: Dict, elementwise: bool = True) -> Optional[np.ndarray]:
        """
        Helper method to apply a core operation and the learnable transformation Wx+b.
        Handles batching, sequence processing, and shape alignment. Ensures float32.
        """
        if not inputs:
            # print("Error (_apply_operation): No inputs provided.") # Debugging
            return None

        learnable_params = params.get('learnable_params', {})
        # W, b dimensions are based on self.feature_dim
        W = learnable_params.get('W', np.eye(self.feature_dim, dtype=np.float32))
        b = learnable_params.get('b', np.zeros(self.feature_dim, dtype=np.float32))

        # Ensure W and b are float32
        W = W.astype(np.float32)
        b = b.astype(np.float32)

        x = inputs[0]
        if x is None:
             print("Error (_apply_operation): Primary input (x) is None.")
             return None
        # Ensure x is numpy array and float32
        if not isinstance(x, np.ndarray):
             print(f"Error (_apply_operation): Primary input (x) is not a numpy array (type: {type(x)}).")
             return None
        if x.dtype != np.float32:
             # print(f"Warning (_apply_operation): Primary input (x) dtype was {x.dtype}, converting to float32.")
             x = x.astype(np.float32)
        if x.ndim != 3:
             print(f"Error (_apply_operation): Primary input (x) has incorrect dimensions {x.ndim}, expected 3 (batch, seq, feat). Shape: {x.shape}")
             return None

        batch_size, seq_len, feature_dim_actual = x.shape

        # Input feature dim must match the environment's feature_dim
        if feature_dim_actual != self.feature_dim:
             print(f"Error (_apply_operation): Input feature dimension {feature_dim_actual} does not match environment feature dimension {self.feature_dim}.")
             return None

        y = inputs[1] if len(inputs) > 1 else None
        if y is not None:
            if not isinstance(y, np.ndarray):
                 print(f"Warning (_apply_operation): Secondary input (y) is not a numpy array (type: {type(y)}). Ignoring y.")
                 y = None
            else:
                 if y.dtype != np.float32:
                      # print(f"Warning (_apply_operation): Secondary input (y) dtype was {y.dtype}, converting to float32.")
                      y = y.astype(np.float32)

                 if y.shape != x.shape:
                    # Attempt broadcasting/tiling for y
                    try:
                        if y.ndim == 3:
                            # Case 1: (batch, 1, feat) -> (batch, seq, feat)
                            if y.shape[0] == batch_size and y.shape[1] == 1 and y.shape[2] == feature_dim_actual:
                                y = np.tile(y, (1, seq_len, 1))
                            # Case 2: (1, seq, feat) -> (batch, seq, feat)
                            elif y.shape[0] == 1 and y.shape[1] == seq_len and y.shape[2] == feature_dim_actual:
                                y = np.tile(y, (batch_size, 1, 1))
                            # Case 3: (1, 1, feat) -> (batch, seq, feat)
                            elif y.shape[0] == 1 and y.shape[1] == 1 and y.shape[2] == feature_dim_actual:
                                y = np.tile(y, (batch_size, seq_len, 1))
                            else:
                                 print(f"Warning (_apply_operation): Secondary input (y) shape {y.shape} incompatible with primary input shape {x.shape} after 3D check. Ignoring y.")
                                 y = None
                        elif y.ndim == 2:
                             # Case 4: (batch, feat) -> (batch, seq, feat)
                             if y.shape[0] == batch_size and y.shape[1] == feature_dim_actual:
                                  y = np.expand_dims(y, 1) # (batch, 1, feat)
                                  y = np.tile(y, (1, seq_len, 1))
                             # Case 5: (seq, feat) -> (batch, seq, feat) - Less common, maybe treat as error?
                             elif y.shape[0] == seq_len and y.shape[1] == feature_dim_actual:
                                  y = np.expand_dims(y, 0) # (1, seq, feat)
                                  y = np.tile(y, (batch_size, 1, 1))
                             else:
                                  print(f"Warning (_apply_operation): Secondary input (y) shape {y.shape} incompatible with primary input shape {x.shape} after 2D check. Ignoring y.")
                                  y = None
                        elif y.ndim == 1:
                             # Case 6: (feat,) -> (batch, seq, feat)
                             if y.shape[0] == feature_dim_actual:
                                  y = np.reshape(y, (1, 1, feature_dim_actual))
                                  y = np.tile(y, (batch_size, seq_len, 1))
                             else:
                                  print(f"Warning (_apply_operation): Secondary input (y) shape {y.shape} incompatible with feature dimension {feature_dim_actual}. Ignoring y.")
                                  y = None
                        else:
                             print(f"Warning (_apply_operation): Secondary input (y) has unsupported dimensions {y.ndim}. Ignoring y.")
                             y = None

                    except Exception as e:
                         print(f"Warning (_apply_operation): Error during y shape alignment: {e}. Ignoring y.")
                         y = None
                 # else: y shape matches x shape, no alignment needed

        # Initialize core_output with correct shape and type
        core_output = np.zeros_like(x, dtype=np.float32)

        try:
            if elementwise:
                # Vectorized elementwise operation if possible
                x_flat = x.reshape(-1, self.feature_dim)
                y_flat = y.reshape(-1, self.feature_dim) if y is not None else None

                results_flat = np.zeros_like(x_flat)
                for i in range(x_flat.shape[0]):
                    x_vec = x_flat[i]
                    y_vec = y_flat[i] if y_flat is not None else None
                    op_result = operation_func(x_vec, y_vec, params) # Operates on feature_dim

                    if op_result is None:
                         op_result = np.zeros(self.feature_dim, dtype=np.float32)
                    elif np.isscalar(op_result):
                         op_result = np.full(self.feature_dim, op_result, dtype=np.float32)
                    elif op_result.shape != (self.feature_dim,):
                         print(f"Warning: Output of elementwise op {operation_func.__name__} has shape {op_result.shape}, expected ({self.feature_dim},). Resizing.")
                         try:
                              op_result = np.resize(op_result, self.feature_dim).astype(np.float32)
                         except Exception as resize_e:
                              print(f"Error resizing output: {resize_e}. Using zeros.")
                              op_result = np.zeros(self.feature_dim, dtype=np.float32)

                    results_flat[i] = op_result.astype(np.float32) # Ensure float32 here

                core_output = results_flat.reshape(batch_size, seq_len, self.feature_dim)

            else: # Sequence operations
                results_list = []
                for b_idx in range(batch_size):
                    x_seq = x[b_idx] # Shape (seq_len, feature_dim)
                    y_seq = y[b_idx] if y is not None else None
                    op_result = operation_func(x_seq, y_seq, params) # Operates on (seq_len, feature_dim)

                    # --- Validate and Align Sequence Op Output ---
                    if op_result is None:
                         print(f"Warning: Sequence op {operation_func.__name__} returned None for batch {b_idx}. Using zeros.")
                         op_result = np.zeros((seq_len, self.feature_dim), dtype=np.float32)
                    elif np.isscalar(op_result):
                         # If scalar, broadcast to (seq_len, feature_dim)
                        #  print(f"Warning: Sequence op {operation_func.__name__} returned scalar {op_result}. Broadcasting.")
                         op_result = np.full((seq_len, self.feature_dim), op_result, dtype=np.float32)
                    elif op_result.shape == (self.feature_dim,):
                         # If op returns single vector (e.g., max), tile it across seq_len
                         op_result = np.tile(op_result.astype(np.float32), (seq_len, 1))
                    elif op_result.shape == (seq_len, self.feature_dim):
                         pass # Correct shape
                    else: # Attempt resize as fallback, log warning
                         print(f"Warning: Output of sequence op {operation_func.__name__} has shape {op_result.shape}, expected ({seq_len}, {self.feature_dim}). Resizing.")
                         try:
                              op_result = np.resize(op_result, (seq_len, self.feature_dim)).astype(np.float32)
                         except Exception as resize_e:
                              print(f"Error resizing output: {resize_e}. Using zeros.")
                              op_result = np.zeros((seq_len, self.feature_dim), dtype=np.float32)
                    # --- End Validation ---

                    results_list.append(op_result.astype(np.float32)) # Ensure float32

                core_output = np.stack(results_list, axis=0) # Stack results into (batch, seq, feat)

        except Exception as core_op_e:
             print(f"Error during core operation execution ({operation_func.__name__}): {core_op_e}")
             traceback.print_exc()
             return None # Return None if core operation fails

        # --- Apply Learnable Transformation (Wx + b) ---
        core_output_flat = core_output.reshape(-1, self.feature_dim)
        try:
             # W is (feat, feat), b is (feat,)
             # core_output_flat is (batch*seq, feat)
             # Result should be (batch*seq, feat)
             transformed_flat = core_output_flat @ W + b # W and b are already float32
        except Exception as transform_e:
             print(f"Error during learnable transformation (Wx+b): {transform_e}")
             traceback.print_exc()
             # Check shapes if error occurs
             print(f"Shapes - core_output_flat: {core_output_flat.shape}, W: {W.shape}, b: {b.shape}")
             return None # Return None if transformation fails

        # Reshape back to (batch, seq_len, feature_dim)
        final_output = transformed_flat.reshape(batch_size, seq_len, self.feature_dim)

        # Handle potential NaN/Inf values
        if not np.all(np.isfinite(final_output)):
             # print(f"Warning: NaN/Inf detected in output of _apply_operation for {operation_func.__name__}. Clamping values.") # Optional warning
             final_output = np.nan_to_num(final_output, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)

        return final_output.astype(np.float32) # Final dtype check


    # --- Core functions (_placeholder_op_core, addition_core, etc.) ---
    # Ensure all core functions return np.float32 arrays or scalars that can be broadcast
    def _placeholder_op_core(self, x, y, params):
         """Core logic for placeholder operation. Returns first input or zeros."""
         print(f"Warning: Using placeholder core for operation (should not happen).")
         if x is not None:
              return x.copy().astype(np.float32)
         return np.zeros(self.feature_dim, dtype=np.float32)


    # --- Arithmetic ---
    def addition_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        res = x + y if y is not None else x
        return res.astype(np.float32)

    def subtraction_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        res = x - y if y is not None else x
        return res.astype(np.float32)

    def multiplication_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        res = x * y if y is not None else x
        return res.astype(np.float32)

    def division_core(self, x, y, params):
        epsilon = 1e-8
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        if y is not None:
            denominator = np.where(np.abs(y) > epsilon, y, epsilon * np.sign(y + epsilon))
            res = x / denominator
        else: # Division by 1 if y is None
            res = x
        return np.nan_to_num(res, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


    # --- Algebra ---
    def exponentiation_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        base = x
        default_exponent = np.full_like(x, 2.0, dtype=np.float32)
        exponent = y if y is not None else default_exponent
        safe_base = np.abs(base) + 1e-6
        safe_exponent = np.clip(exponent, -5, 5)
        try:
            result = np.power(safe_base, safe_exponent)
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
            print(f"Error in exponentiation_core: {e}")
            return np.zeros_like(x, dtype=np.float32)


    def root_extraction_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        val = x
        default_root = np.full_like(x, 2.0, dtype=np.float32)
        root_val = y if y is not None else default_root
        epsilon = 1e-6
        safe_val = np.abs(val) + epsilon
        safe_root = np.clip(root_val, -10, 10)
        safe_root = np.where(np.abs(safe_root) > epsilon, safe_root, epsilon * np.sign(safe_root + epsilon))
        inv_root = 1.0 / safe_root
        try:
            result = np.power(safe_val, inv_root)
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
            print(f"Error in root_extraction_core: {e}")
            return np.zeros_like(x, dtype=np.float32)


    # --- Calculus (Sequence Operations) ---
    def derivative_core(self, x_seq, y_seq, params):
        # x_seq shape: (seq_len, feature_dim)
        if x_seq is None:
             print("Warning (derivative_core): x_seq is None.")
             return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        if x_seq.shape[0] < 2:
            return np.zeros_like(x_seq, dtype=np.float32)
        try:
            grad = np.gradient(x_seq, axis=0)
            return np.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
            print(f"Error in derivative_core: {e}")
            return np.zeros_like(x_seq, dtype=np.float32)


    def integral_core(self, x_seq, y_seq, params):
        # x_seq shape: (seq_len, feature_dim)
        if x_seq is None:
            print("Warning (integral_core): x_seq is None.")
            return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        try:
            cumsum = np.cumsum(x_seq, axis=0)
            return np.nan_to_num(cumsum, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
            print(f"Error in integral_core: {e}")
            return np.zeros_like(x_seq, dtype=np.float32)


    # --- Transforms (Sequence Operations) ---
    def fourier_transform_core(self, x_seq, y_seq, params):
        if x_seq is None:
             print("Warning (fourier_transform_core): x_seq is None.")
             return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        seq_len, feat_dim = x_seq.shape

        try:
            # Use numpy's FFT
            fft_result = np_fft.fft(x_seq, axis=0) # Shape: (seq_len, feat_dim), dtype=complex

            # Represent complex output in real-valued feature space
            output = np.zeros_like(x_seq, dtype=np.float32)
            if feat_dim >= 2:
                 real_part = np.real(fft_result)
                 imag_part = np.imag(fft_result)
                 # Interleave real and imaginary parts, handling odd feature dim
                 half_dim = feat_dim // 2
                 output[:, 0:half_dim*2:2] = real_part[:, :half_dim]
                 output[:, 1:half_dim*2+1:2] = imag_part[:, :half_dim]
                 if feat_dim % 2 == 1: # Handle odd feature dimension
                      output[:, -1] = real_part[:, half_dim] # Store last real part
            else: # feat_dim == 1
                 output[:, 0] = np.abs(fft_result)[:, 0]

            return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
             print(f"Error in fourier_transform_core: {e}")
             traceback.print_exc()
             return np.zeros_like(x_seq, dtype=np.float32)


    def inverse_fourier_transform_core(self, x_seq, y_seq, params):
        if x_seq is None:
             print("Warning (inverse_fourier_transform_core): x_seq is None.")
             return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        seq_len, feat_dim = x_seq.shape

        try:
            # Reconstruct complex representation from real-valued features
            complex_repr = np.zeros((seq_len, feat_dim // 2 + feat_dim % 2), dtype=np.complex128)
            if feat_dim >= 1:
                 complex_repr.real = x_seq[:, 0::2] # Real parts from even indices
            if feat_dim >= 2:
                 complex_repr.imag[:, :feat_dim//2] = x_seq[:, 1::2] # Imaginary parts from odd indices

            # Compute IFFT using numpy
            ifft_result = np_fft.ifft(complex_repr, n=seq_len, axis=0) # Specify output length n=seq_len

            # Return the real part, ensure shape matches original feature dim
            real_ifft = np.real(ifft_result)
            output = np.zeros_like(x_seq, dtype=np.float32)
            cols_to_copy = min(real_ifft.shape[1], feat_dim)
            output[:, :cols_to_copy] = real_ifft[:, :cols_to_copy] # Copy back, potentially padding with zeros if ifft output is smaller

            return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
             print(f"Error in inverse_fourier_transform_core: {e}")
             traceback.print_exc()
             return np.zeros_like(x_seq, dtype=np.float32)


    # --- Functional Analysis (Sequence Operation) ---
    def convolution_core(self, x_seq, y_seq, params):
        if x_seq is None:
             print("Warning (convolution_core): x_seq is None.")
             return np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        seq_len, feat_dim = x_seq.shape
        result = np.zeros_like(x_seq, dtype=np.float32)

        kernel_source = None
        if y_seq is not None:
             if y_seq.ndim == 2 and y_seq.shape[1] == feat_dim and y_seq.shape[0] > 0:
                  kernel_source = y_seq
             else:
                  print(f"Warning (convolution_core): y_seq shape {y_seq.shape} incompatible. Using default kernel.")

        if kernel_source is not None:
             kernel_len = kernel_source.shape[0]
             for f in range(feat_dim):
                  try:
                       kernel = kernel_source[:, f]
                       result[:, f] = signal.convolve(x_seq[:, f], kernel, mode='same', method='auto')
                  except Exception as e:
                       print(f"Convolution error (using y_seq) feature {f}: {e}")
                       result[:, f] = x_seq[:, f] # Fallback
        else:
             kernel_size = min(5, seq_len)
             if kernel_size > 0:
                  try:
                       kernel = signal.windows.gaussian(kernel_size, std=1)
                       kernel /= np.sum(kernel)
                       for f in range(feat_dim):
                            result[:, f] = signal.convolve(x_seq[:, f], kernel, mode='same', method='auto')
                  except Exception as e:
                       print(f"Default convolution error: {e}")
                       result = x_seq.copy()
             else:
                  result = x_seq.copy()

        return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)


    # --- Set/Logic Operations (Elementwise Fuzzy Logic Placeholders) ---
    def set_union_core(self, x, y, params): # Fuzzy OR (max)
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        res = np.maximum(x, y) if y is not None else x
        return res.astype(np.float32)

    def set_intersection_core(self, x, y, params): # Fuzzy AND (min)
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        res = np.minimum(x, y) if y is not None else x
        return res.astype(np.float32)

    def set_complement_core(self, x, y, params): # Fuzzy NOT (1 - x)
        if x is None: return np.ones(self.feature_dim, dtype=np.float32)
        clipped_x = np.clip(x, 0.0, 1.0)
        return (1.0 - clipped_x).astype(np.float32)

    def logical_and_core(self, x, y, params):
        return self.set_intersection_core(x, y, params)

    def logical_or_core(self, x, y, params):
        return self.set_union_core(x, y, params)

    def logical_not_core(self, x, y, params):
        return self.set_complement_core(x, y, params)


    # --- Geometry (Elementwise Operations on feature vectors) ---
    def rotation_core(self, x_vec, y_vec, params):
         if x_vec is None: return np.zeros(self.feature_dim, dtype=np.float32)
         if self.feature_dim < 2: return x_vec.astype(np.float32)

         theta = y_vec[0] if y_vec is not None and y_vec.size > 0 else np.pi / 4.0
         try: theta = float(theta)
         except (ValueError, TypeError): theta = np.pi / 4.0

         try:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
            result = np.copy(x_vec)
            result[:2] = rot_matrix @ x_vec[:2]
            return result.astype(np.float32)
         except Exception as e:
             print(f"Error in rotation_core: {e}")
             return x_vec.astype(np.float32)


    def reflection_core(self, x_vec, y_vec, params):
        if x_vec is None: return np.zeros(self.feature_dim, dtype=np.float32)
        if self.feature_dim < 2: return x_vec.astype(np.float32)

        theta = y_vec[0] if y_vec is not None and y_vec.size > 0 else np.pi / 2.0
        try: theta = float(theta)
        except (ValueError, TypeError): theta = np.pi / 2.0

        try:
            cos_2t, sin_2t = np.cos(2 * theta), np.sin(2 * theta)
            refl_matrix = np.array([[cos_2t, sin_2t], [sin_2t, -cos_2t]], dtype=np.float32)
            result = np.copy(x_vec)
            result[:2] = refl_matrix @ x_vec[:2]
            return result.astype(np.float32)
        except Exception as e:
            print(f"Error in reflection_core: {e}")
            return x_vec.astype(np.float32)


    def translation_core(self, x_vec, y_vec, params):
        if x_vec is None: return np.zeros(self.feature_dim, dtype=np.float32)
        default_translation = np.full(self.feature_dim, 0.1, dtype=np.float32)
        translation_vector = y_vec if y_vec is not None else default_translation

        try:
            if translation_vector.shape != x_vec.shape:
                 translation_vector = np.resize(translation_vector, x_vec.shape)
            return (x_vec + translation_vector).astype(np.float32)
        except Exception as e:
            print(f"Error in translation_core: {e}")
            return x_vec.astype(np.float32)


    def distance_core(self, x_vec, y_vec, params):
        # Returns a scalar (broadcast by _apply_operation)
        if x_vec is None or y_vec is None: return 0.0
        try:
            if y_vec.shape != x_vec.shape: y_vec = np.resize(y_vec, x_vec.shape)
            dist = np.linalg.norm(x_vec - y_vec)
            return np.float32(dist) if np.isfinite(dist) else np.float32(0.0)
        except Exception as e:
            print(f"Error in distance_core: {e}")
            return np.float32(0.0)


    # --- Linear Algebra (Elementwise Operations on feature vectors) ---
    def scalar_multiplication_core(self, x_vec, y_vec, params):
        if x_vec is None: return np.zeros(self.feature_dim, dtype=np.float32)
        scalar = y_vec[0] if y_vec is not None and y_vec.size > 0 else 1.0
        try: scalar = float(scalar)
        except (ValueError, TypeError): scalar = 1.0
        return (x_vec * scalar).astype(np.float32)


    def inner_product_core(self, x_vec, y_vec, params):
        # Returns a scalar (broadcast by _apply_operation)
        if x_vec is None or y_vec is None: return 0.0
        try:
            if y_vec.shape != x_vec.shape: y_vec = np.resize(y_vec, x_vec.shape)
            dot = np.dot(x_vec, y_vec)
            return np.float32(dot) if np.isfinite(dot) else np.float32(0.0)
        except Exception as e:
            print(f"Error in inner_product_core: {e}")
            return np.float32(0.0)


    # Inside MathSelfPlayEnv class in math_env.py

    def matrix_multiplication_core(self, x, y, params):
        # Placeholder: Elementwise product
        # print("Warning (matrix_multiplication_core): Using elementwise product as placeholder.") # <--- THIS LINE CAUSES THE WARNING
        if x is None: return np.zeros_like(y, dtype=np.float32) if y is not None else np.zeros(self.feature_dim, dtype=np.float32)
        if y is None: return x.astype(np.float32)
        try:
            if y.shape != x.shape: y = np.resize(y, x.shape)
            return (x * y).astype(np.float32)
        except Exception as e:
             print(f"Error in matrix_multiplication_core (placeholder): {e}")
             return x.astype(np.float32)



    # --- Number Theory (Elementwise, requires integer interpretation) ---
    def gcd_core(self, x, y, params):
        if x is None or y is None:
             return np.ones(self.feature_dim, dtype=np.float32)
        try:
            x_int = np.round(np.abs(x)).astype(int)
            y_int = np.round(np.abs(y)).astype(int)
            if y_int.shape != x_int.shape: y_int = np.resize(y_int, x_int.shape)
            gcd_func = np.vectorize(math.gcd)
            result = gcd_func(x_int, y_int)
            return result.astype(np.float32)
        except Exception as e:
            print(f"Error in gcd_core: {e}")
            return np.ones_like(x, dtype=np.float32)


    def modulo_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim, dtype=np.float32)
        if y is None:
             # print("Warning (modulo_core): y is None. Returning x.") # Reduce noise
             return x.astype(np.float32)

        epsilon = 1e-8
        try:
            if y.shape != x.shape: y = np.resize(y, x.shape)
            divisor = np.where(np.abs(y) > epsilon, y, epsilon * np.sign(y + epsilon))
            result = np.fmod(x, divisor)
            return np.nan_to_num(result, nan=0.0).astype(np.float32)
        except Exception as e:
            print(f"Error in modulo_core: {e}")
            return x.astype(np.float32)


    def factorial_core(self, x, y, params):
        if x is None: return np.ones(self.feature_dim, dtype=np.float32)
        try:
            x_int = np.clip(np.round(x), 0, 15).astype(int) # Limit range
            result = special.factorial(x_int, exact=False)
            return np.nan_to_num(result, nan=1.0, posinf=1e6).astype(np.float32)
        except Exception as e:
            print(f"Error in factorial_core: {e}")
            return np.ones_like(x, dtype=np.float32)


    def binomial_coefficient_core(self, x, y, params):
        if x is None or y is None: return np.zeros(self.feature_dim, dtype=np.float32)
        try:
            n = np.clip(np.round(x), 0, 30).astype(int)
            k = np.clip(np.round(y), 0, 30).astype(int)
            if k.shape != n.shape: k = np.resize(k, n.shape)
            result = special.comb(n, k, exact=False)
            return np.nan_to_num(result, nan=0.0, posinf=1e6).astype(np.float32)
        except Exception as e:
            print(f"Error in binomial_coefficient_core: {e}")
            return np.zeros_like(x, dtype=np.float32)


    # --- Analysis/Order (Supremum/Infimum are sequence ops, Measure is placeholder) ---
    def supremum_core(self, x_seq, y_seq, params):
        # Returns shape (feature_dim,) -> broadcast by _apply_operation
        if x_seq is None or x_seq.size == 0:
             return np.zeros(self.feature_dim, dtype=np.float32)
        try:
            result = np.max(x_seq, axis=0)
            return result.astype(np.float32) if np.all(np.isfinite(result)) else np.zeros(self.feature_dim, dtype=np.float32)
        except Exception as e:
            print(f"Error in supremum_core: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


    def infimum_core(self, x_seq, y_seq, params):
        # Returns shape (feature_dim,) -> broadcast by _apply_operation
        if x_seq is None or x_seq.size == 0:
             return np.zeros(self.feature_dim, dtype=np.float32)
        try:
            result = np.min(x_seq, axis=0)
            return result.astype(np.float32) if np.all(np.isfinite(result)) else np.zeros(self.feature_dim, dtype=np.float32)
        except Exception as e:
            print(f"Error in infimum_core: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)


    def measure_core(self, x_seq, y_seq, params):
        # Placeholder: L1 norm over sequence. Returns scalar -> broadcast by _apply_operation
        if x_seq is None or x_seq.size == 0: return np.float32(0.0)
        try:
            measure = np.sum(np.abs(x_seq))
            return np.float32(measure) if np.isfinite(measure) else np.float32(0.0)
        except Exception as e:
            print(f"Error in measure_core: {e}")
            return np.float32(0.0)


    # --- Information Theory (Sequence Operation) ---
    def entropy_core(self, x_seq, y_seq, params):
        # Returns a scalar -> broadcast by _apply_operation
        if x_seq is None or x_seq.size == 0: return np.float32(0.0)

        try:
            flat_seq = x_seq.flatten()
            min_val, max_val = np.min(flat_seq), np.max(flat_seq)
            range_val = max_val - min_val
            if range_val < 1e-8: return np.float32(0.0) # Entropy is 0 if all values are the same

            normalized_seq = (flat_seq - min_val) / range_val
            num_bins = 10
            hist, _ = np.histogram(normalized_seq, bins=num_bins, range=(0, 1))
            total_counts = hist.sum()
            if total_counts == 0: return np.float32(0.0)

            probabilities = hist[hist > 0] / total_counts
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return np.float32(entropy) if np.isfinite(entropy) else np.float32(0.0)
        except Exception as e:
            print(f"Error in entropy_core: {e}")
            traceback.print_exc()
            return np.float32(0.0)

    def close(self):
        """Clean up any resources (if needed)."""
        pass # No specific resources to close in this version

