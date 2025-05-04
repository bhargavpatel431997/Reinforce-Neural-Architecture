# Reinforce Neural Architecture (For Sequence Task Optimization)

This project focuses on enabling AI to autonomously construct optimal neural architectures from scratch. By leveraging a defined set of fundamental mathematical and tensor operations, the AI is guided via Reinforcement Learning (Self-Play PPO) to discover and create entirely new neural network structures tailored for sequence-to-sequence tasks.

# Main Objective

The goal is to allow the AI to explore beyond human-designed architectures—potentially uncovering novel neural network designs that may have been previously overlooked by researchers, specifically by composing operations with learnable parameters.

# Conceptual Basis: Fundamental Math

*(**Disclaimer:** The following table lists broad mathematical concepts. The actual operations implemented in the environment are adapted versions suitable for tensor computation within a PyTorch graph, often incorporating learnable parameters. See the "Implemented Operations" section below for the precise list and behavior.)*

| Field                | Operation             | Notation                | Function Form              |
| :------------------- | :-------------------- | :---------------------- | :------------------------- |
| **Arithmetic**       | Addition              | `x + y`                 | `f(x,y)=x+y`               |
| **Arithmetic**       | Subtraction           | `x - y`                 | `f(x,y)=x-y`               |
| **Arithmetic**       | Multiplication        | `x * y`                 | `f(x,y)=x·y`               |
| **Arithmetic**       | Division              | `x / y`                 | `f(x,y)=x/y`               |
| **Algebra**          | Exponentiation        | `x^y`                   | `f(x,y)=x^y`               |
| **Algebra**          | Root Extraction       | `y√x`                   | `f(x,y)=x^(1/y)`           |
| **Algebra**          | Logarithm             | `log x`                 | `f(x)=log(x)`              |
| **Transforms**       | Fourier Transform     | `ℱ{f}(ω)`             | `∫f(t)e^{-iωt}dt`          |
| **Functional Analysis**| Convolution           | `f * g`                 | `∫f(τ)g(t−τ)dτ`            |
| **Linear Algebra**   | Matrix Multiplication | `A·B`                   | matrix product             |
| **Linear Algebra**   | Inner Product         | `⟨u,v⟩`                 | sum of component products  |
| **Number Theory**    | Modulo                | `x mod n`               | remainder of division      |
| **Order/Analysis**   | Supremum / Infimum    | `sup S` / `inf S`       | least upper / greatest lower bound |
| **Geometry**         | Translation           | `T_v(P)`                | `P + v`                    |
| **Geometry**         | Scaling               | `S_k(P)`                | multiplies distance by k   |
| *(Other fields like Calculus, Set Theory, Logic, etc., are currently not directly implemented as placeable nodes)* |                       |                         |                            |

# Core Idea

The system treats neural network architecture generation as a game played on a grid. Two players (controlled by the **same** PPO agent policy) take turns placing computational nodes (mathematical/tensor operations from a predefined list) onto the grid. The goal is to build a computational graph that, when evaluated on a sequence task dataset (e.g., string reversal), achieves low Mean Squared Error loss between the graph's output sequence and the target sequence.

# Components

1.  **Environment (`MathSelfPlayEnv` in `math_env.py`)**
    *   **The Game Board:** Provides a `grid_size` x `grid_size` space where the network graph is built.
    *   **Sequence Task:**
        *   Generates batches of sequence data (e.g., string reversal, integer addition) using `_generate_sequence_data`.
        *   Embeds input characters/tokens into tensors of `feature_dim` using a `char_to_point` dictionary.
        *   Holds the current batch of input tensors (`self.current_inputs` shape `(B, S, F)`) and target tensors (`self.target_outputs` shape `(B, S, F)`).
        *   *(Note: Unlike a previous MNIST setup, fixed input/output adapter layers are not used here; embedding is part of data generation).*
    *   **Graph Host:** Contains the `ComputationalGraph` instance (`self.graph`) being built on the specified PyTorch device.
    *   **State Representation:** Provides the current state to the agent as an observation dictionary, primarily containing the `board` state (a multi-channel NumPy tensor representing node types, player ownership, and pointer location on the grid).
    *   **Evaluation:** When requested by the agent (implicitly during `env.step`), it evaluates the *current* graph structure:
        1.  Performs a `forward_pass` through the constructed `self.graph` using the `current_inputs`.
        2.  Calculates `MSELoss` between the graph's output tensor and the `target_outputs`.
    *   **Reward Calculation:** Computes the reward signal based on the change in evaluation loss compared to the previous step, penalizing invalid moves, evaluation failures, and large losses, while rewarding successful node placement and loss improvement.

2.  **Agent (`SelfPlayPPOAgent` in `PPO_agent.py`)**
    *   **The Player:** A single PPO agent instance is used for *both* Player 1 and Player 2.
    *   **Policy Network:** Contains the `SelfPlayTransformerPPONetwork` which decides the actions based on the board state.
    *   **Actions:** At each turn, the agent chooses:
        *   `operation_id`: Which implemented mathematical/tensor operation node to place (See "Implemented Operations" below).
        *   `placement_strategy`: Where to place the node relative to the previous node or the input node (Up, Down, Left, Right, RelativeToInput).
    *   **Learning:** Updates its policy network using experiences (state, action, reward, etc.) gathered from both players' turns stored in a replay buffer (`self.buffer`). The PPO loss includes policy loss, value loss, and an entropy bonus for exploration.

3.  **Policy Network (`SelfPlayTransformerPPONetwork` in `PPO_agent.py`)**
    *   **The Brain:** A Transformer-based network designed to process the grid-based graph state.
    *   **Input:** Takes the `board` state tensor from the environment observation.
    *   **Processing:** Uses an encoder (e.g., `EnhancedGraphTransformerEncoder` likely using Graph Attention) to process the spatial and operational information on the grid.
    *   **Output:**
        *   Probability distributions over `operation_id` and `placement_strategy`.
        *   A value estimate `V(s)` for the current state (used by PPO).

4.  **Computational Graph (`ComputationalGraph` in `math_env.py`)**
    *   **The Generated Network:** Represents the neural network architecture being built as a Directed Acyclic Graph (DAG) of `MathNode` objects.
    *   **Nodes (`MathNode`):** Each node represents a computation step and contains:
        *   `op_id`: An integer identifying the specific operation.
        *   `op_info`: Dictionary holding operation details (name, arity, core function).
        *   **`learnable_param`**: An optional `torch.nn.Parameter` tensor. This single parameter holds the learnable weights/biases *for this specific node*. Its shape and interpretation depend on the node's operation and `learnable_role`.
        *   **`learnable_role`**: A string indicating how `learnable_param` is used (e.g., `'operand_y'`, `'bias'`, `'scale'`).
        *   **Binary Operations (`arity=2`, `learnable_role='operand_y'`)**:
            *   The `learnable_param` typically has shape `(feature_dim, feature_dim)`.
            *   **Element-wise Ops (Add, Sub, Mul, Div, Pow, Root, Mod):** Use the *diagonal* of the `learnable_param` for the operation (e.g., `x + diag(y_learnable)`).
            *   **MatMul:** Performs `x @ y_learnable`. Adapts if `x` has feature dim 1 (treats `y_learnable` as an expansion layer using its first column).
            *   **Conv1D:** Reshapes `y_learnable` into a 1x1 convolutional kernel `(F, F, 1)`.
            *   **InnerProd:** Uses the *diagonal* of `y_learnable` and computes `sum(x * diag(y_learnable), dim=-1)`, reducing feature dimension to 1.
        *   **Unary Operations (`arity=1`)**:
            *   May have a `learnable_param` if a `learnable_param_role` (like `'bias'` or `'scale'`) is defined.
            *   Example: `LayerNorm` uses a bias vector `(F,)`. `Translate` uses a bias vector. `Scale` uses a scale vector `(F,)`.
            *   Most unary ops (Tanh, ReLU, Log, FFT, Reductions) do not have learnable parameters by default.
        *   `inputs`: A list of parent `MathNode` objects providing input.
        *   `position`: Tuple `(row, col)` on the grid.
        *   `player_id`: Which player placed the node.
    *   **Evaluation (`forward_pass`):**
        *   Executes the graph topologically using Kahn's algorithm.
        *   **Input Combination:** If a node has multiple inputs, their output tensors are combined. Handles broadcasting/expansion for inputs with differing sequence lengths (S vs 1) or feature dimensions (F vs 1) before averaging. Fails node evaluation if dimensions are fundamentally incompatible (e.g., different batch sizes, non-broadcastable feature dims).
        *   Applies each node's core PyTorch function (`*_core`) using the combined input and the node's specific `learnable_param`.
    *   **Serialization (`serialize_graph`):** Converts the graph structure (nodes, positions, connections, op_id, learnable_role) into a saveable list of dictionaries (JSON format). **Does not save the learned parameter *values***.

# Implemented Operations (`_potential_ops` in `MathSelfPlayEnv`)

The agent can choose from the following operations to place as nodes. Tensor shapes are typically `(Batch, Sequence, FeatureDim)` denoted as `(B, S, F)`. `y` refers to the node's `learnable_param`.

| ID | Name      | Arity | Input Shape | Learnable Param (y) Role & Shape | Output Shape | Description                                                                 |
| :- | :-------- | :---- | :---------- | :------------------------------- | :----------- | :-------------------------------------------------------------------------- |
| 0  | Add       | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `x + diag(y)`                                                               |
| 1  | Sub       | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `x - diag(y)`                                                               |
| 2  | Mul       | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `x * diag(y)` (Element-wise)                                                |
| 3  | Div       | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `x / (diag(y) + eps)`                                                       |
| 4  | Pow       | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `relu(x+eps) ^ clamp(diag(y), -10, 10)`                                     |
| 5  | Root      | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `relu(x+eps) ^ clamp(1 / (diag(y)+eps), -20, 20)`                            |
| 6  | MatMul    | 2     | `(B,S,F)` or `(B,S,1)` | `operand_y` (`F,F`) | `(B,S,F)`    | `x @ y` (Adapts if input feature=1, uses first col of `y` to expand to `F`) |
| 7  | InnerProd | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,1)`    | `sum(x * diag(y), dim=-1, keepdim=True)`                                    |
| 8  | Mod       | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | `fmod(x, abs(diag(y))+eps)`                                                 |
| 9  | FFT_Mag   | 1     | `(B,S,F)`   | None                             | `(B,S,F)`    | `abs(fft(x, dim=1))`                                                        |
| 10 | IFFT_Plc  | 1     | `(B,S,F)`   | None                             | `(B,S,F)`    | Placeholder (Identity op, as phase is lost from FFT_Mag)                    |
| 11 | Conv1D    | 2     | `(B,S,F)`   | `operand_y` (`F,F`)              | `(B,S,F)`    | 1x1 Convolution using `y` reshaped to `(F,F,1)` kernel                       |
| 12 | Tanh      | 1     | `(B,S,F)`   | None                             | `(B,S,F)`    | `tanh(x)`                                                                   |
| 13 | ReLU      | 1     | `(B,S,F)`   | None                             | `(B,S,F)`    | `relu(x)`                                                                   |
| 14 | Sigmoid   | 1     | `(B,S,F)`   | None                             | `(B,S,F)`    | `sigmoid(x)`                                                                |
| 15 | Log       | 1     | `(B,S,F)`   | None                             | `(B,S,F)`    | `log(relu(x) + eps)` (Natural Log)                                          |
| 16 | LayerNorm | 1     | `(B,S,F)`   | `bias` (`F,`)                    | `(B,S,F)`    | Layer Normalization over feature dim, with optional learnable bias        |
| 17 | Supremum  | 1     | `(B,S,F)`   | None                             | `(B,1,F)`    | `max(x, dim=1, keepdim=True)`                                               |
| 18 | Infimum   | 1     | `(B,S,F)`   | None                             | `(B,1,F)`    | `min(x, dim=1, keepdim=True)`                                               |
| 19 | Mean      | 1     | `(B,S,F)`   | None                             | `(B,1,F)`    | `mean(x, dim=1, keepdim=True)`                                              |
| 20 | Translate | 1     | `(B,S,F)`   | `bias` (`F,`)                    | `(B,S,F)`    | `x + y_bias`                                                                |
| 21 | Scale     | 1     | `(B,S,F)`   | `scale` (`F,`)                   | `(B,S,F)`    | `x * y_scale`                                                               |

# The Self-Play Loop & Learning

The training process (`agent.train`) involves simulating many episodes where the agent builds and evaluates graphs:

```mermaid
graph TD
    A[Start Episode] --> B(Reset Env: Get Seq Batch, Empty Graph)
    B --> C{Episode Not Done?}
    C -- Yes --> D[Get Current Player & Board State]
    D --> E["Agent's Policy Network Processes State"]
    E --> F["Agent Samples Action (Op ID, Placement)"]
    F --> G[Env Executes Action: Place Node, Connect]
    G --> H{Valid Move? DAG, Grid, Connections}
    H -- No --> I[Penalize Agent, Revert Move]
    H -- Yes --> J["Env Evaluates Current Graph on Seq Batch"]
    J --> K["Env Calculates Loss (MSE) and Reward Signal"]
    K --> L["Store (State, Action, Reward, Done, LogProb, Value) in Buffer"]
    L --> M{Buffer Full?}
    M -- Yes --> N[Agent Updates Policy via PPO using Buffer Data]
    N --> O[Clear Buffer]
    M -- No --> O
    O --> P[Switch Player, Update State]
    P --> C
    C -- No --> Q[End Episode]
    Q --> R{Track Best Graph based on Eval Loss?}
    R -- Yes --> S[Save Best Graph Structure JSON]
    R -- No --> T[Repeat for Next Episode]
    S --> T
    I --> P
