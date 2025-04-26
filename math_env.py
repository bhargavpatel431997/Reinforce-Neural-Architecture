# -*- coding: utf-8 -*-
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import copy
from typing import List, Dict, Tuple, Any, Optional
import random
# import string # No longer needed for data gen
import math
from scipy import signal
from scipy.fft import fft, ifft
from scipy import special # For comb (binomial coefficient)
import traceback # For detailed error printing
import json

# --- NEW IMPORTS ---
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# --- END NEW IMPORTS ---


# --- MathNode Class ---
# ... (MathNode class remains the same) ...
class MathNode:
    """
    Represents a mathematical operation node in the computational graph.
    Each node applies a specific mathematical function to its inputs,
    followed by a learnable affine transformation (W @ result + b).
    """
    def __init__(self, op_id: int, name: str, inputs=None, parameters=None, feature_dim=64, player_id=0): # Default feature_dim
        self.op_id = op_id             # ID of the operation from the operations list
        self.name = name               # Human-readable name of the operation
        self.inputs = inputs or []     # List of input nodes
        self.parameters = parameters or {}  # Fixed parameters (can be empty if not used by agent)
        self.output = None            # Output tensor after evaluation
        self.position = None          # (row, col) position in the graph
        self.output_shape = None      # Shape of the output tensor
        self.player_id = player_id    # Which player placed this node (1 or 2)

        # Learnable parameters for the transformation W @ result + b
        # Initialized when the node is added to the environment
        self.learnable_params = {
            'W': np.random.randn(feature_dim, feature_dim) * 0.1, # Small random weights
            'b': np.zeros(feature_dim)                            # Zero bias
        }
        # Store a unique ID for easier tracking if needed
        self.unique_id = id(self)

    def add_input(self, node):
        """Add an input node to this node."""
        if node not in self.inputs:
            self.inputs.append(node)

    def set_parameters(self, parameters):
        """Set fixed parameters for this operation."""
        self.parameters = parameters

    def __repr__(self):
        input_ids = [inp.unique_id for inp in self.inputs] # Use unique_id for clarity
        return f"MathNode(id={self.unique_id}, name={self.name}, op={self.op_id}, pos={self.position}, player={self.player_id}, inputs={input_ids})"


# --- ComputationalGraph Class ---
# ... (ComputationalGraph class remains the same, including serialize_graph) ...
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
        for node in self.nodes:
            if node.unique_id not in visited:
                if is_cyclic_util(node.unique_id):
                    return False
        return True

    def topological_sort(self) -> List[MathNode]:
        """Return nodes in topological order."""
        visited = set()
        topo_order = []

        def visit(node_id):
            node = self.get_node_by_id(node_id)
            if not node or node_id in visited: return
            visited.add(node_id)
            for inp in node.inputs:
                visit(inp.unique_id)
            topo_order.append(node)

        # Ensure all nodes are considered, especially disconnected ones if any
        nodes_to_visit = list(self.nodes_by_id.keys())
        while nodes_to_visit:
            node_id = nodes_to_visit.pop(0)
            if node_id not in visited:
                 visit(node_id)
        return topo_order

    def forward_pass(self, input_tensor: np.ndarray, operations: Dict[int, Dict], input_embedding: nn.Module) -> Optional[np.ndarray]: # Added input_embedding
        """
        Perform forward pass. Assumes a single input tensor and a single designated output node.
        Applies input embedding to the initial data.
        """
        if not self.input_node:
            return None
        if not self.output_node:
             return None

        # Reset outputs
        for node in self.nodes:
            node.output = None
            node.output_shape = None

        # --- Apply Input Embedding ---
        try:
            # input_tensor shape: (batch, 1, 784) - assuming this format from _get_next_batch
            batch_size, seq_len, input_dim_actual = input_tensor.shape
            # Flatten for embedding layer: (batch * seq_len, input_dim_actual)
            input_flat = torch.tensor(input_tensor.reshape(-1, input_dim_actual), dtype=torch.float32)
            embedded_flat = input_embedding(input_flat) # Output: (batch * seq_len, feature_dim)
            # Reshape back to sequence format: (batch, seq_len, feature_dim)
            embedded_input_np = embedded_flat.reshape(batch_size, seq_len, -1).detach().numpy()

            self.input_node.output = embedded_input_np # Use embedded input
            self.input_node.output_shape = embedded_input_np.shape
        except Exception as e:
            print(f"Error applying input embedding: {e}")
            traceback.print_exc()
            return None
        # --- End Input Embedding ---


        sorted_nodes = self.topological_sort()
        final_output = None # This will be the output of the designated self.output_node

        for node in sorted_nodes:
            if node == self.input_node:
                continue # Already assigned (with embedded input)

            # Collect inputs (ensure they are available)
            node_inputs = []
            inputs_ready = True
            if not node.inputs:
                 if node != self.input_node:
                      inputs_ready = False
            else:
                 for inp_node in node.inputs:
                      if inp_node.output is None:
                           inputs_ready = False
                           break
                      node_inputs.append(inp_node.output)

            if not inputs_ready:
                 node.output = None
                 continue

            # Apply operation (using the _apply_operation helper from the env)
            if node.op_id in operations:
                try:
                    op_info = operations[node.op_id]
                    apply_func = op_info['apply']
                    core_func = op_info['core']
                    is_elementwise = op_info.get('elementwise', True)

                    op_params = {
                        'inputs': node_inputs,
                        'learnable_params': node.learnable_params,
                        **node.parameters
                    }

                    node.output = apply_func(node_inputs, core_func, op_params, is_elementwise)

                    if node.output is not None:
                        node.output_shape = node.output.shape

                except Exception as e:
                    print(f"Error evaluating node {node.name} ({node.unique_id}) op {node.op_id}: {str(e)}")
                    traceback.print_exc()
                    node.output = None
            else:
                 node.output = None


            # Check if this is the designated output node
            if node == self.output_node:
                final_output = node.output

        # Return the raw output of the final graph node (before projection)
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
                # "name": node.name, # Optional: include name for readability
                "position": node.position, # Tuple (row, col)
                "player_id": node.player_id,
                "input_ids": [inp.unique_id for inp in node.inputs] # List of IDs of input nodes
            }
            serialized_nodes.append(node_data)
        return serialized_nodes

# --- MathSelfPlayEnv Class ---
class MathSelfPlayEnv(gym.Env):
    """
    Gym environment for self-play graph construction for digit classification.
    Players take turns placing learnable math nodes on a grid.
    Reward is based on classification loss and expansion penalty.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 4}

    # --- Reward Weights ---
    ACCURACY_PENALTY_WEIGHT = 5.0 # Increased weight for classification loss
    EXPANSION_PENALTY = 0.05      # Penalty for increasing grid dimensions

    def __init__(self, grid_size=5, max_steps=20, feature_dim=64, batch_size=64, dataset_path='./data'): # Changed defaults
        super().__init__()

        self.grid_size = grid_size
        self.max_nodes = grid_size * grid_size
        self.max_steps = max_steps # Max turns in a game
        self.feature_dim = feature_dim # Intermediate feature dimension for graph nodes
        self.input_feature_dim = 28 * 28 # MNIST image dimension
        self.num_classes = 10 # MNIST classes
        self.batch_size = batch_size # Batch size for evaluation
        self.dataset_path = dataset_path

        # --- Dataset Loading ---
        self.train_loader = None
        self.train_iter = None
        self._load_dataset()

        # --- Input/Output Layers (Part of Env Evaluation Harness) ---
        self.input_embedding = nn.Linear(self.input_feature_dim, self.feature_dim)
        self.output_projection = nn.Linear(self.feature_dim, self.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        # --- End Dataset/Eval Layers ---


        # --- Define Operations (same as before) ---
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
        _ids_to_remove = {10, 11, 12, 13, 18, 22, 23, 32}
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
             is_elementwise = not (name in ["Derivative", "Integral", "Fourier Transform",
                                            "Inverse Fourier Transform", "Convolution", "Entropy",
                                            "Measure", "Supremum", "Infimum"])
             self.operations_impl[op_id] = {
                 'core': core_func, 'apply': self._apply_operation,
                 'elementwise': is_elementwise, 'name': name
             }
        vector_add_new_id = self._original_to_new_id_map.get(24)
        add_new_id = self._original_to_new_id_map.get(0)
        if vector_add_new_id is not None and add_new_id is not None:
             self.operations_impl[vector_add_new_id]['core'] = self.addition_core
        # --- End Operations Definition ---


        # --- Action Space ---
        self.num_placement_strategies = 5
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
        self.last_loss: float = float('inf')
        self.steps_taken: int = 0
        self.current_inputs: Optional[np.ndarray] = None # Will store flattened images (batch, 1, 784)
        self.target_outputs: Optional[np.ndarray] = None # Will store labels (batch,)
        # self.char_to_point = None # No longer needed

    # --- NEW METHOD: Load Dataset ---
    def _load_dataset(self):
        print(f"Loading MNIST dataset from {self.dataset_path}...")
        try:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)), # MNIST mean/std
                # Flatten handled in _get_next_batch
            ])
            train_dataset = torchvision.datasets.MNIST(
                root=self.dataset_path, train=True, download=True, transform=transform
            )
            # Use a reasonably large batch size for the DataLoader
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=2, persistent_workers=True)
            self.train_iter = iter(self.train_loader)
            print("MNIST dataset loaded.")
        except Exception as e:
            print(f"Error loading MNIST dataset: {e}")
            print("Please ensure torchvision is installed and data can be downloaded/accessed.")
            raise e

    # --- NEW METHOD: Get Batch ---
    def _get_next_batch(self):
        try:
            images, labels = next(self.train_iter)
        except StopIteration:
            # Epoch finished, create new iterator
            self.train_iter = iter(self.train_loader)
            images, labels = next(self.train_iter)

        # Flatten images and reshape for sequence length 1
        # Shape: (batch_size, 1, 784)
        images_flat = images.view(images.size(0), -1).unsqueeze(1)

        self.current_inputs = images_flat.numpy().astype(np.float32)
        self.target_outputs = labels.numpy() # Shape: (batch_size,)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.graph = ComputationalGraph()
        self.current_player = 1
        self.pointer_location = None
        self.last_loss = 10.0 # Initial loss guess (log(10) is ~2.3, maybe higher)
        self.steps_taken = 0

        # --- Get first batch of data ---
        self._get_next_batch()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: Dict[str, int]):
        """Take a turn in the game by placing a node."""
        if self.graph is None:
            raise RuntimeError("Environment needs to be reset before stepping.")

        operation_id = action['operation_id']
        placement_strategy = action['placement_strategy']

        terminated = False
        reward = 0.0
        info = {'error': ''}
        new_node = None

        previous_pointer_location = self.pointer_location
        prev_max_row = self.graph.max_row
        prev_max_col = self.graph.max_col

        try:
            # --- 1. Determine Target Position ---
            target_row, target_col = -1, -1
            is_first_move = (len(self.graph.nodes) == 0)

            if is_first_move:
                if placement_strategy == 0: target_row, target_col = 0, 0
                else: raise ValueError("Invalid placement strategy for the first move. Must be 0.")
            else:
                if self.pointer_location is None: raise RuntimeError("Pointer location is None after the first move.")
                pointer_row, pointer_col = self.pointer_location
                if placement_strategy == 1: target_row, target_col = pointer_row - 1, pointer_col
                elif placement_strategy == 2: target_row, target_col = pointer_row, pointer_col + 1
                elif placement_strategy == 3: target_row, target_col = pointer_row + 1, pointer_col
                elif placement_strategy == 4: target_row, target_col = pointer_row, pointer_col - 1
                elif placement_strategy == 0: raise ValueError(f"Invalid placement strategy: {placement_strategy} (0 is only for first move)")
                else: raise ValueError(f"Unknown placement strategy: {placement_strategy}")

            # --- 2. Validate Position ---
            if not (0 <= target_row < self.grid_size and 0 <= target_col < self.grid_size):
                raise ValueError(f"Invalid move: Position ({target_row},{target_col}) is off-grid ({self.grid_size}x{self.grid_size}).")
            if self.graph.get_node_at(target_row, target_col) is not None:
                occupied_node = self.graph.get_node_at(target_row, target_col)
                raise ValueError(f"Invalid move: Position ({target_row},{target_col}) is occupied by {occupied_node.name}.")

            # --- 3. Create and Add Node ---
            if not (0 <= operation_id < self.num_operations):
                 raise ValueError(f"Invalid operation_id: {operation_id} (should be 0-{self.num_operations-1})")

            op_name = self.operations_impl[operation_id]['name']
            node_name = f"{op_name}_{len(self.graph.nodes)}_P{self.current_player}"
            # Pass the intermediate feature_dim to the node
            new_node = MathNode(
                op_id=operation_id, name=node_name,
                feature_dim=self.feature_dim, player_id=self.current_player
            )
            self.graph.add_node(new_node, target_row, target_col)

            # --- 4. Connect Node ---
            if not is_first_move and previous_pointer_location:
                prev_node = self.graph.get_node_at(*previous_pointer_location)
                if prev_node:
                    self.graph.connect_nodes(prev_node.unique_id, new_node.unique_id)
                else:
                    print(f"Warning: Node not found at previous pointer {previous_pointer_location}. Connecting from input.")
                    if self.graph.input_node and self.graph.input_node != new_node:
                         self.graph.connect_nodes(self.graph.input_node.unique_id, new_node.unique_id)

            # --- 5. Update Pointer ---
            self.pointer_location = (target_row, target_col)

            # --- 6. Check DAG ---
            if not self.graph.is_valid_dag():
                self.graph.remove_node(new_node)
                self.pointer_location = previous_pointer_location
                new_node = None
                raise ValueError("Invalid move: Created a cycle.")

            # --- 7. Evaluate Graph & Calculate Reward ---
            self.graph.set_output_node(new_node)
            current_loss = float('inf')
            # Pass the input embedding layer to forward_pass
            eval_output_np = self.graph.forward_pass(self.current_inputs, self.operations_impl, self.input_embedding)

            # --- Reward Calculation Logic (Classification) ---
            if eval_output_np is not None:
                # Expected output shape: (batch, 1, feature_dim)
                expected_shape = (self.batch_size, 1, self.feature_dim)
                if eval_output_np.shape == expected_shape:
                    try:
                        # Convert to tensor, squeeze seq dim, apply projection
                        eval_output_tensor = torch.tensor(eval_output_np.squeeze(1), dtype=torch.float32) # Shape: (batch, feature_dim)
                        logits = self.output_projection(eval_output_tensor) # Shape: (batch, num_classes)

                        # Calculate Cross-Entropy loss
                        target_tensor = torch.tensor(self.target_outputs, dtype=torch.long) # Shape: (batch,)
                        loss_tensor = self.criterion(logits, target_tensor)
                        loss = loss_tensor.item() # Get scalar loss

                        if np.isnan(loss) or np.isinf(loss):
                            current_loss = 100.0 # Assign large finite loss for bad numerics
                            reward = -1.0 # Strong penalty for NaN/Inf
                            info['error'] = "Evaluation resulted in NaN/Inf loss."
                        else:
                            current_loss = loss
                            # Reward = Improvement - Accuracy Penalty - Expansion Penalty
                            finite_last_loss = self.last_loss if np.isfinite(self.last_loss) else 100.0
                            improvement = finite_last_loss - current_loss
                            accuracy_penalty = self.ACCURACY_PENALTY_WEIGHT * current_loss
                            reward = improvement - accuracy_penalty
                            # Optional: Add accuracy bonus?
                            # preds = torch.argmax(logits, dim=1)
                            # accuracy = (preds == target_tensor).float().mean().item()
                            # reward += accuracy * 0.1 # Small bonus for accuracy

                    except Exception as loss_calc_e:
                         current_loss = 100.0
                         reward = -0.8 # Penalty for error during loss calculation
                         info['error'] = f"Error during loss calculation: {loss_calc_e}"
                         traceback.print_exc()

                else:
                    # Shape mismatch penalty
                    current_loss = self.last_loss + 10.0 if np.isfinite(self.last_loss) else 100.0 + 10.0
                    reward = -0.5 # Penalty for shape mismatch
                    info['error'] = f"Output shape mismatch: Expected {expected_shape}, Got {eval_output_np.shape}"
            else:
                # Forward pass failed
                current_loss = self.last_loss + 20.0 if np.isfinite(self.last_loss) else 100.0 + 20.0
                reward = -0.7 # Penalty for evaluation failure
                info['error'] = "Graph evaluation failed or produced None output."

            # --- Apply Expansion Penalty ---
            expanded_grid = (target_row > prev_max_row) or (target_col > prev_max_col)
            if expanded_grid:
                reward -= self.EXPANSION_PENALTY
                info['grid_expanded'] = True

            self.last_loss = current_loss
            # --- End Reward Calculation ---

        except ValueError as e:
            reward = -1.0
            info['error'] = str(e)
            self.pointer_location = previous_pointer_location

        except Exception as e:
            reward = -2.0
            info['error'] = f"Unexpected error: {str(e)}"
            traceback.print_exc()
            terminated = True
            if new_node and new_node in self.graph.nodes:
                 self.graph.remove_node(new_node)
            self.pointer_location = previous_pointer_location

        # --- 8. Update Step Counter and Switch Player ---
        self.steps_taken += 1
        self.current_player = 3 - self.current_player

        # --- 9. Check Termination Conditions ---
        if self.steps_taken >= self.max_steps:
            terminated = True
            info['termination_reason'] = 'max_steps_reached'
        if len(self.graph.nodes) >= self.max_nodes:
             terminated = True
             info['termination_reason'] = 'max_nodes_reached'

        # --- 10. Prepare return values ---
        observation = self._get_observation()
        if np.isnan(reward) or np.isinf(reward):
             reward = -1.0

        truncated = False
        info['last_loss'] = self.last_loss

        return observation, reward, terminated, truncated, info

    # --- _get_observation, _get_info, render remain the same ---
    def _get_observation(self):
        """Construct the observation dictionary."""
        board_shape = self.observation_space['board'].shape
        board = np.zeros(board_shape, dtype=np.float32)

        if self.graph is None: # Should not happen after reset
             return {
                'board': board,
                'current_player': self.current_player,
                'steps_taken': self.steps_taken
             }

        # Channel indices based on updated observation space definition
        op_channel_offset = 0 # Channels 0 to num_operations-1 (now 0-32)
        input_channel_idx = self.num_operations # Channel 33
        player1_channel_idx = self.num_operations + 1 # Channel 34
        player2_channel_idx = self.num_operations + 2 # Channel 35
        pointer_channel_idx = self.num_operations + 3 # Channel 36

        for node in self.graph.nodes:
            if node.position is None: continue # Should not happen for added nodes
            r, c = node.position
            if 0 <= r < self.grid_size and 0 <= c < self.grid_size:
                # Mark operation type or input node
                if node.op_id == -1: # Input node
                    board[r, c, input_channel_idx] = 1.0
                elif 0 <= node.op_id < self.num_operations: # Check against new range
                    board[r, c, op_channel_offset + node.op_id] = 1.0

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

        return {
            'board': board,
            'current_player': self.current_player,
            'steps_taken': self.steps_taken
        }

    def _get_info(self):
        """Return auxiliary information about the environment state."""
        return {
            'last_loss': self.last_loss,
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
        print(f"Last Eval Loss: {self.last_loss:.4f}")
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


    # --- _generate_char_sequence_data removed ---

    # --- _apply_operation and _core functions remain the same ---
    # Note: They now operate on self.feature_dim (e.g., 64)
    def _apply_operation(self, inputs: List[np.ndarray], operation_func: callable, params: Dict, elementwise: bool = True) -> Optional[np.ndarray]:
        """
        Helper method to apply a core operation and the learnable transformation Wx+b.
        Handles batching, sequence processing, and shape alignment.
        """
        if not inputs:
            return None

        learnable_params = params.get('learnable_params', {})
        # W, b dimensions are based on self.feature_dim (intermediate dim)
        W = learnable_params.get('W', np.eye(self.feature_dim))
        b = learnable_params.get('b', np.zeros(self.feature_dim))

        x = inputs[0]
        if x is None:
             print("Error (_apply_operation): Primary input is None.")
             return None
        batch_size, seq_len, feature_dim_actual = x.shape

        # Input feature dim must match the environment's intermediate dim
        if feature_dim_actual != self.feature_dim:
             print(f"Error (_apply_operation): Input feature dimension {feature_dim_actual} does not match environment feature dimension {self.feature_dim}.")
             return None

        y = inputs[1] if len(inputs) > 1 else None
        if y is not None:
            if y.shape != x.shape:
                try:
                    if y.shape[0] == batch_size and y.shape[2] == feature_dim_actual and y.shape[1] == 1:
                        y = np.tile(y, (1, seq_len, 1))
                    elif y.shape[0] == 1 and y.shape[1] == seq_len and y.shape[2] == feature_dim_actual:
                        y = np.tile(y, (batch_size, 1, 1))
                    elif y.shape == (batch_size, feature_dim_actual):
                         y = np.expand_dims(y, 1)
                         y = np.tile(y, (1, seq_len, 1))
                    elif y.shape == (feature_dim_actual,):
                         y = np.reshape(y, (1, 1, feature_dim_actual))
                         y = np.tile(y, (batch_size, seq_len, 1))
                    else:
                        y = None
                except Exception as e:
                     print(f"Warning (_apply_operation): Error during y shape alignment: {e}. Ignoring y.")
                     y = None

        core_output = np.zeros_like(x) # Shape (batch, seq=1, feature_dim)

        try:
            if elementwise:
                for b_idx in range(batch_size):
                    for t_idx in range(seq_len): # seq_len is likely 1 here
                        x_vec = x[b_idx, t_idx]
                        y_vec = y[b_idx, t_idx] if y is not None else None
                        op_result = operation_func(x_vec, y_vec, params) # Operates on feature_dim

                        if op_result is None:
                             op_result = np.zeros(self.feature_dim)
                        elif np.isscalar(op_result):
                             op_result = np.full(self.feature_dim, op_result, dtype=np.float32)
                        elif op_result.shape != (self.feature_dim,):
                             op_result = np.resize(op_result, self.feature_dim).astype(np.float32)

                        core_output[b_idx, t_idx] = op_result
            else: # Sequence operations (might be less relevant if seq_len=1)
                for b_idx in range(batch_size):
                    x_seq = x[b_idx]
                    y_seq = y[b_idx] if y is not None else None
                    op_result = operation_func(x_seq, y_seq, params) # Operates on (seq_len, feature_dim)

                    if op_result is None:
                         op_result = np.zeros((seq_len, self.feature_dim))
                    elif np.isscalar(op_result):
                        op_result = np.full((seq_len, self.feature_dim), op_result, dtype=np.float32)
                    elif op_result.shape == (self.feature_dim,):
                        op_result = np.tile(op_result, (seq_len, 1))
                    elif op_result.shape == (seq_len, self.feature_dim):
                        pass
                    else:
                         try:
                              op_result = np.resize(op_result, (seq_len, self.feature_dim)).astype(np.float32)
                         except Exception as resize_e:
                              op_result = np.zeros((seq_len, self.feature_dim))

                    core_output[b_idx] = op_result

        except Exception as core_op_e:
             print(f"Error during core operation execution ({operation_func.__name__}): {core_op_e}")
             traceback.print_exc()
             return None

        core_output_flat = core_output.reshape(-1, self.feature_dim)
        try:
             W_np = np.asarray(W)
             b_np = np.asarray(b)
             transformed_flat = core_output_flat @ W_np + b_np
        except Exception as transform_e:
             print(f"Error during learnable transformation (Wx+b): {transform_e}")
             traceback.print_exc()
             return None

        final_output = transformed_flat.reshape(batch_size, seq_len, self.feature_dim)

        if np.any(np.isnan(final_output)) or np.any(np.isinf(final_output)):
             final_output = np.nan_to_num(final_output, nan=0.0, posinf=1e6, neginf=-1e6, copy=False)

        return final_output.astype(np.float32)

    # --- Core functions (_placeholder_op_core, addition_core, etc.) ---
    # --- These remain the same as before, operating on self.feature_dim ---
    # Placeholder for unimplemented operations (should only be hit if getattr fails)
    def _placeholder_op_core(self, x, y, params):
         """Core logic for placeholder operation. Returns first input or zeros."""
         print(f"Warning: Using placeholder core for operation (should not happen).")
         if x is not None:
              return x.copy()
         if hasattr(x, 'shape'):
              # Need to know if sequence or vector expected
              # Assume vector output if elementwise, sequence otherwise
              # This logic is tricky without knowing context, default to vector
              return np.zeros(self.feature_dim)
         else:
              return np.zeros(self.feature_dim)

    # --- Arithmetic ---
    def addition_core(self, x, y, params):
        if y is not None: return x + y
        return x

    def subtraction_core(self, x, y, params):
        if y is not None: return x - y
        return x

    def multiplication_core(self, x, y, params):
        if y is not None: return x * y
        return x

    def division_core(self, x, y, params):
        epsilon = 1e-8
        if y is not None:
            denominator = np.where(np.abs(y) > epsilon, y, epsilon * np.sign(y + epsilon))
            return x / denominator
        return x # Division by 1 if y is None

    # --- Algebra ---
    def exponentiation_core(self, x, y, params):
        base = x
        exponent = y if y is not None else 2.0 # Default to square
        base = np.abs(base) + 1e-6 # Ensure base > 0
        exponent = np.clip(exponent, -5, 5) # Clip exponent range
        try:
            result = np.power(base, exponent)
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6) # Clamp result
        except Exception:
            return np.zeros_like(x)

    def root_extraction_core(self, x, y, params):
        val = x
        root_val = y if y is not None else 2.0 # Default to square root
        epsilon = 1e-6
        val = np.abs(val) + epsilon # Ensure val > 0
        root_val = np.clip(root_val, -10, 10) # Clip root range
        safe_root = np.where(np.abs(root_val) > epsilon, root_val, epsilon * np.sign(root_val + epsilon))
        # Avoid division by zero in exponent
        inv_root = 1.0 / safe_root
        try:
            result = np.power(val, inv_root)
            return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6) # Clamp result
        except Exception:
            return np.zeros_like(x)

    # --- Calculus (Sequence Operations) ---
    def derivative_core(self, x_seq, y_seq, params):
        # x_seq shape: (seq_len, feature_dim)
        if x_seq is None or x_seq.shape[0] < 2:
            # Return shape consistent with input seq_len
            seq_len = x_seq.shape[0] if x_seq is not None else 1
            return np.zeros((seq_len, self.feature_dim))
        try:
            return np.gradient(x_seq, axis=0)
        except Exception:
            return np.zeros_like(x_seq)

    def integral_core(self, x_seq, y_seq, params):
        # x_seq shape: (seq_len, feature_dim)
        if x_seq is None:
            return np.zeros((1, self.feature_dim)) # Assume seq_len=1 if input is None
        try:
            return np.cumsum(x_seq, axis=0)
        except Exception:
            return np.zeros_like(x_seq)

    # --- Transforms (Sequence Operations) ---
    def fourier_transform_core(self, x_seq, y_seq, params):
        seq_len = x_seq.shape[0] if x_seq is not None else 1
        if x_seq is None: return np.zeros((seq_len, self.feature_dim))
        try:
            fft_result = np.fft.fft(x_seq, axis=0)
            if self.feature_dim >= 2:
                 real_part = np.real(fft_result)
                 imag_part = np.imag(fft_result)
                 output = np.zeros_like(x_seq)
                 num_complex_features = fft_result.shape[1]
                 for i in range(num_complex_features):
                      real_idx, imag_idx = 2*i, 2*i + 1
                      if real_idx < self.feature_dim: output[:, real_idx] = real_part[:, i]
                      if imag_idx < self.feature_dim: output[:, imag_idx] = imag_part[:, i]
                 return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)
            else:
                 mag = np.abs(fft_result)
                 return np.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=-1e6)
        except Exception as e:
             print(f"Error in FFT: {e}")
             return np.zeros_like(x_seq)

    def inverse_fourier_transform_core(self, x_seq, y_seq, params):
        seq_len = x_seq.shape[0] if x_seq is not None else 1
        if x_seq is None: return np.zeros((seq_len, self.feature_dim))
        try:
            num_complex_features = self.feature_dim // 2 + self.feature_dim % 2
            complex_repr = np.zeros((x_seq.shape[0], num_complex_features), dtype=np.complex128)
            for i in range(num_complex_features):
                 real_idx, imag_idx = 2*i, 2*i + 1
                 real_part = x_seq[:, real_idx] if real_idx < self.feature_dim else 0
                 imag_part = x_seq[:, imag_idx] if imag_idx < self.feature_dim else 0
                 complex_repr[:, i] = real_part + 1j * imag_part

            ifft_result = np.fft.ifft(complex_repr, axis=0)
            real_ifft = np.real(ifft_result)

            output = np.zeros_like(x_seq)
            cols_to_copy = min(real_ifft.shape[1], self.feature_dim)
            output[:, :cols_to_copy] = real_ifft[:, :cols_to_copy]
            if cols_to_copy < self.feature_dim and cols_to_copy > 0:
                 last_feature = output[:, cols_to_copy-1:cols_to_copy]
                 output[:, cols_to_copy:] = np.tile(last_feature, (1, self.feature_dim - cols_to_copy))

            return np.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
        except Exception as e:
             print(f"Error in IFFT: {e}")
             return np.zeros_like(x_seq)

    # --- Functional Analysis (Sequence Operation) ---
    def convolution_core(self, x_seq, y_seq, params):
        seq_len = x_seq.shape[0] if x_seq is not None else 1
        if x_seq is None: return np.zeros((seq_len, self.feature_dim))
        result = np.zeros_like(x_seq)

        if y_seq is not None:
            for f in range(self.feature_dim):
                try:
                    kernel = y_seq[:, f]
                    result[:, f] = np.convolve(x_seq[:, f], kernel, mode='same')
                except Exception as e:
                     print(f"Convolution error feature {f}: {e}")
                     result[:, f] = x_seq[:, f] # Fallback
        else:
            kernel_size = min(5, seq_len)
            if kernel_size > 0:
                 try:
                    kernel = signal.windows.gaussian(kernel_size, std=1)
                    kernel /= np.sum(kernel) # Normalize
                    for f in range(self.feature_dim):
                          result[:, f] = np.convolve(x_seq[:, f], kernel, mode='same')
                 except Exception as e:
                      print(f"Default convolution error: {e}")
                      result = x_seq # Fallback
            else:
                 result = x_seq # No convolution if seq_len is 0

        return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)

    # --- Set/Logic Operations (Elementwise Fuzzy Logic Placeholders) ---
    def set_union_core(self, x, y, params):
        if y is not None: return np.maximum(x, y)
        return x
    def set_intersection_core(self, x, y, params):
        if y is not None: return np.minimum(x, y)
        return x
    def set_complement_core(self, x, y, params):
        return 1.0 - np.clip(x, 0.0, 1.0)
    def logical_and_core(self, x, y, params):
        return self.set_intersection_core(x, y, params)
    def logical_or_core(self, x, y, params):
        return self.set_union_core(x, y, params)
    def logical_not_core(self, x, y, params):
        return self.set_complement_core(x, y, params)

    # --- Geometry (Elementwise Operations) ---
    def rotation_core(self, x_vec, y_vec, params):
         if x_vec is None: return np.zeros(self.feature_dim)
         if self.feature_dim < 2: return x_vec
         theta = y_vec[0] if y_vec is not None and y_vec.size > 0 else np.pi / 4.0
         try:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
            rotated_part = rot_matrix @ x_vec[:2]
            result = np.copy(x_vec)
            result[:2] = rotated_part
            return result
         except Exception:
             return x_vec

    def reflection_core(self, x_vec, y_vec, params):
        if x_vec is None: return np.zeros(self.feature_dim)
        if self.feature_dim < 2: return x_vec
        theta = y_vec[0] if y_vec is not None and y_vec.size > 0 else np.pi / 2.0
        try:
            cos_2t, sin_2t = np.cos(2 * theta), np.sin(2 * theta)
            refl_matrix = np.array([[cos_2t, sin_2t], [sin_2t, -cos_2t]])
            reflected_part = refl_matrix @ x_vec[:2]
            result = np.copy(x_vec)
            result[:2] = reflected_part
            return result
        except Exception:
            return x_vec

    def translation_core(self, x_vec, y_vec, params):
        if x_vec is None: return np.zeros(self.feature_dim)
        translation_vector = y_vec if y_vec is not None else np.full(self.feature_dim, 0.1)
        try:
            if translation_vector.shape != x_vec.shape:
                 translation_vector = np.resize(translation_vector, x_vec.shape)
            return x_vec + translation_vector
        except Exception:
            return x_vec

    def distance_core(self, x_vec, y_vec, params):
        if x_vec is None or y_vec is None: return 0.0
        try:
            if y_vec.shape != x_vec.shape:
                 y_vec = np.resize(y_vec, x_vec.shape)
            dist = np.linalg.norm(x_vec - y_vec)
            return dist if np.isfinite(dist) else 0.0
        except Exception:
            return 0.0

    # --- Linear Algebra ---
    def scalar_multiplication_core(self, x_vec, y_vec, params):
        if x_vec is None: return np.zeros(self.feature_dim)
        scalar = y_vec[0] if y_vec is not None and y_vec.size > 0 else 1.0
        return x_vec * scalar

    def inner_product_core(self, x_vec, y_vec, params):
        if x_vec is None or y_vec is None: return 0.0
        try:
            if y_vec.shape != x_vec.shape:
                 y_vec = np.resize(y_vec, x_vec.shape)
            dot = np.dot(x_vec, y_vec)
            return dot if np.isfinite(dot) else 0.0
        except Exception:
            return 0.0

    def matrix_multiplication_core(self, x, y, params):
        # Placeholder: Elementwise product
        if x is None: return np.zeros_like(y) if y is not None else np.zeros(self.feature_dim)
        if y is None: return x
        try:
            if y.shape != x.shape:
                 y = np.resize(y, x.shape)
            return x * y
        except Exception:
             return x

    # --- Number Theory (Elementwise, requires integer interpretation) ---
    def gcd_core(self, x, y, params):
        if x is None or y is None: return np.ones_like(x) if x is not None else np.ones(self.feature_dim)
        try:
            x_int = np.round(x).astype(int)
            y_int = np.round(y).astype(int)
            gcd_func = np.vectorize(math.gcd)
            result = gcd_func(x_int, y_int)
            return result.astype(np.float32)
        except Exception as e:
            print(f"GCD error: {e}")
            return np.ones_like(x)

    def modulo_core(self, x, y, params):
        if x is None: return np.zeros(self.feature_dim)
        if y is None: return x
        epsilon = 1e-8
        try:
            divisor = np.where(np.abs(y) > epsilon, y, epsilon * np.sign(y + epsilon))
            return np.fmod(x, divisor)
        except Exception:
            return x

    def factorial_core(self, x, y, params):
        if x is None: return np.ones(self.feature_dim)
        try:
            x_int = np.clip(np.round(x), 0, 15).astype(int)
            result = special.factorial(x_int, exact=False)
            return np.nan_to_num(result, nan=1.0, posinf=1e6).astype(np.float32)
        except Exception as e:
            print(f"Factorial error: {e}")
            return np.ones_like(x)

    def binomial_coefficient_core(self, x, y, params):
        if x is None or y is None: return np.zeros(self.feature_dim)
        try:
            n = np.clip(np.round(x), 0, 30).astype(int)
            k = np.clip(np.round(y), 0, 30).astype(int)
            valid_k = np.where(k <= n, k, -1)
            result = special.comb(n, valid_k, exact=False)
            return np.nan_to_num(result, nan=0.0, posinf=1e6).astype(np.float32)
        except Exception as e:
            print(f"Binomial Coefficient error: {e}")
            return np.zeros_like(x)

    # --- Analysis/Order (Supremum/Infimum are sequence ops, Measure is placeholder) ---
    def supremum_core(self, x_seq, y_seq, params):
        seq_len = x_seq.shape[0] if x_seq is not None else 1
        if x_seq is None or x_seq.size == 0: return np.zeros(self.feature_dim)
        try:
            result = np.max(x_seq, axis=0)
            return result if np.all(np.isfinite(result)) else np.zeros(self.feature_dim)
        except Exception:
            return np.zeros(self.feature_dim)

    def infimum_core(self, x_seq, y_seq, params):
        seq_len = x_seq.shape[0] if x_seq is not None else 1
        if x_seq is None or x_seq.size == 0: return np.zeros(self.feature_dim)
        try:
            result = np.min(x_seq, axis=0)
            return result if np.all(np.isfinite(result)) else np.zeros(self.feature_dim)
        except Exception:
            return np.zeros(self.feature_dim)

    def measure_core(self, x_seq, y_seq, params):
        if x_seq is None: return 0.0
        try:
            measure = np.sum(np.abs(x_seq))
            return measure if np.isfinite(measure) else 0.0
        except Exception:
            return 0.0

    # --- Information Theory (Sequence Operation) ---
    def entropy_core(self, x_seq, y_seq, params):
        seq_len = x_seq.shape[0] if x_seq is not None else 1
        if x_seq is None or x_seq.size == 0: return 0.0
        try:
            flat_seq = x_seq.flatten()
            min_val, max_val = np.min(flat_seq), np.max(flat_seq)
            if max_val > min_val:
                normalized_seq = (flat_seq - min_val) / (max_val - min_val)
            else:
                normalized_seq = np.zeros_like(flat_seq)

            num_bins = 10
            hist, _ = np.histogram(normalized_seq, bins=num_bins, range=(0, 1))
            total_counts = hist.sum()
            if total_counts == 0: return 0.0
            probabilities = hist[hist > 0] / total_counts
            entropy = -np.sum(probabilities * np.log2(probabilities))
            return entropy if np.isfinite(entropy) else 0.0
        except Exception as e:
            print(f"Entropy calculation error: {e}")
            return 0.0

    # --- Removed Placeholder Core Function Definitions ---


# --- Example Usage (Illustrative - Needs Agent Script) ---
if __name__ == "__main__":
    # This part is just for basic environment testing, not full training
    print("Testing MNIST Environment Setup...")
    try:
        env = MathSelfPlayEnv(
            grid_size=4,
            max_steps=5,
            feature_dim=64, # Intermediate dimension
            batch_size=16,
            dataset_path='./mnist_data' # Specify path
        )
        print("Environment created.")
        obs, info = env.reset()
        print("Environment reset successful.")
        print("Observation keys:", obs.keys())
        print("Board shape:", obs['board'].shape)
        print("Input data shape:", env.current_inputs.shape) # Should be (batch, 1, 784)
        print("Target data shape:", env.target_outputs.shape) # Should be (batch,)
        print("Initial Info:", info)
        env.render()

        # Test a random step
        action = env.action_space.sample()
        if env.steps_taken == 0: action['placement_strategy'] = 0
        print("\nTaking a random step:", action)
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step result: Reward={reward:.4f}, Term={term}, Trunc={trunc}")
        print("Info:", info)
        env.render()

    except Exception as e:
        print("\n--- Error during environment test ---")
        print(e)
        traceback.print_exc()
        print("------------------------------------")

