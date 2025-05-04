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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import string
import random

def create_char_mapping_csv(output_path="character_mapping.csv"):
    """
    Create a CSV file with mappings from characters to 2D coordinates.
    Characters are mapped to points on a circle plus some randomness.
    
    Args:
        output_path: Path to save the CSV file
    """
    # Characters to include
    chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + string.punctuation + " "
    
    # Create dataframe
    df = pd.DataFrame(columns=['character', 'x', 'y'])
    
    # Generate coordinates for each character
    for i, char in enumerate(chars):
        # Position on circle with some noise
        angle = 2 * np.pi * i / len(chars)
        radius = 0.5 + 0.1 * np.random.randn()  # Add some radius noise
        
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        
        # Add to dataframe
        df = pd.concat([df, pd.DataFrame({'character': [char], 'x': [x], 'y': [y]})], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Character mapping saved to {output_path}")
    
    return df

def load_char_mapping(csv_path, normalize=True):
    """
    Load character mapping from CSV file.
    
    Args:
        csv_path: Path to CSV file with character mappings
        normalize: Whether to normalize coordinates to [-1, 1]
        
    Returns:
        Dictionary mapping characters to 2D vectors
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"CSV file not found at {csv_path}. Creating default mapping.")
        create_char_mapping_csv(csv_path)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Create mapping dictionary
    char_to_vector = {}
    
    # Normalize coordinates if requested
    if normalize:
        x_min, x_max = df['x'].min(), df['x'].max()
        y_min, y_max = df['y'].min(), df['y'].max()
        
        # Avoid division by zero
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        for _, row in df.iterrows():
            char = row['character']
            x = (row['x'] - x_min) / x_range * 2 - 1  # Scale to [-1, 1]
            y = (row['y'] - y_min) / y_range * 2 - 1  # Scale to [-1, 1]
            char_to_vector[char] = np.array([x, y], dtype=np.float32)
    else:
        for _, row in df.iterrows():
            char = row['character']
            x = row['x']
            y = row['y']
            char_to_vector[char] = np.array([x, y], dtype=np.float32)
    
    return char_to_vector

def sentence_to_vector_sequence(sentence, char_to_vector, max_length=20):
    """
    Convert a sentence to a sequence of 2D vectors.
    
    Args:
        sentence: Input sentence
        char_to_vector: Dictionary mapping characters to vectors
        max_length: Maximum sequence length
        
    Returns:
        Numpy array of shape (max_length, 2)
    """
    # Initialize output array
    sequence = np.zeros((max_length, 2), dtype=np.float32)
    
    # Convert characters to vectors
    for i, char in enumerate(sentence):
        if i >= max_length:
            break
            
        if char in char_to_vector:
            sequence[i] = char_to_vector[char]
        else:
            # Unknown character - use default value
            sequence[i] = np.array([0.0, 0.0], dtype=np.float32)
    
    return sequence

def batch_to_vector_sequences(sentences, char_to_vector, max_length=20):
    """
    Convert a batch of sentences to a batch of vector sequences.
    
    Args:
        sentences: List of sentences
        char_to_vector: Dictionary mapping characters to vectors
        max_length: Maximum sequence length
        
    Returns:
        Numpy array of shape (batch_size, max_length, 2)
    """
    batch_size = len(sentences)
    batch = np.zeros((batch_size, max_length, 2), dtype=np.float32)
    
    for i, sentence in enumerate(sentences):
        batch[i] = sentence_to_vector_sequence(sentence, char_to_vector, max_length)
    
    return batch

def visualize_char_mapping(char_to_vector, title="Character Mapping", save_path=None):
    """
    Visualize the character mapping as a 2D plot.
    
    Args:
        char_to_vector: Dictionary mapping characters to 2D vectors
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(12, 10))
    
    # Plot each character
    for char, vec in char_to_vector.items():
        plt.plot(vec[0], vec[1], 'o', markersize=8)
        plt.text(vec[0], vec[1], char, fontsize=12)
    
    plt.grid(True)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def visualize_sentence(sentence, char_to_vector, title=None, save_path=None):
    """
    Visualize a sentence as a path in 2D space.
    
    Args:
        sentence: Input sentence
        char_to_vector: Dictionary mapping characters to 2D vectors
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(12, 10))
    
    # Get vector sequence
    sequence = sentence_to_vector_sequence(sentence, char_to_vector, max_length=len(sentence))
    
    # Plot vectors and connecting lines
    x = sequence[:len(sentence), 0]
    y = sequence[:len(sentence), 1]
    
    plt.plot(x, y, '-o', markersize=8)
    
    # Add character labels
    for i, char in enumerate(sentence):
        plt.text(x[i], y[i], char, fontsize=12)
    
    # Add numbering to show sequence
    for i in range(len(sentence)):
        plt.text(x[i], y[i]+0.05, f"{i}", fontsize=8, color='red')
    
    plt.grid(True)
    plt.title(title or f'Sequence Visualization: "{sentence}"')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def save_training_data(sentences, char_to_vector, output_path, max_length=20):
    """
    Save training data (sentences converted to vector sequences) to a numpy file.
    
    Args:
        sentences: List of sentences
        char_to_vector: Dictionary mapping characters to vectors
        output_path: Path to save the numpy file
        max_length: Maximum sequence length
    """
    # Convert sentences to vector sequences
    data = batch_to_vector_sequences(sentences, char_to_vector, max_length)
    
    # Save to numpy file
    np.save(output_path, data)
    print(f"Training data saved to {output_path}")

def process_text_file(input_file, char_to_vector, max_length=20, output_file=None):
    """
    Process a text file and convert it to vector sequences.
    
    Args:
        input_file: Path to text file
        char_to_vector: Dictionary mapping characters to vectors
        max_length: Maximum sequence length
        output_file: Path to save the numpy file (if None, return data)
    
    Returns:
        Numpy array of shape (n_sentences, max_length, 2) if output_file is None
    """
    # Read text file
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into sentences (simple splitting by period)
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    
    # Convert to vector sequences
    data = batch_to_vector_sequences(sentences, char_to_vector, max_length)
    
    if output_file:
        np.save(output_file, data)
        print(f"Processed {len(sentences)} sentences, saved to {output_file}")
        return None
    else:
        return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Character Mapping and Sequence Conversion Utility")
    
    parser.add_argument('--create_mapping', action='store_true', help='Create a new character mapping CSV')
    parser.add_argument('--csv_path', type=str, default='character_mapping.csv', help='Path to character mapping CSV')
    parser.add_argument('--visualize_mapping', action='store_true', help='Visualize character mapping')
    parser.add_argument('--visualize_sentence', type=str, help='Sentence to visualize')
    parser.add_argument('--normalize', action='store_true', help='Normalize coordinates to [-1, 1]')
    parser.add_argument('--process_file', type=str, help='Process a text file')
    parser.add_argument('--output', type=str, help='Output path for processed data')
    parser.add_argument('--max_length', type=int, default=20, help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Create mapping if requested
    if args.create_mapping:
        create_char_mapping_csv(args.csv_path)
    
    # Load character mapping
    char_to_vector = load_char_mapping(args.csv_path, normalize=args.normalize)
    
    # Visualize mapping if requested
    if args.visualize_mapping:
        visualize_char_mapping(char_to_vector, save_path=args.output)
    
    # Visualize sentence if requested
    if args.visualize_sentence:
        visualize_sentence(args.visualize_sentence, char_to_vector, save_path=args.output)
    
    # Process text file if requested
    if args.process_file:
        process_text_file(args.process_file, char_to_vector, args.max_length, args.output)

# Example usage in code:
"""
# Create a character mapping CSV
create_char_mapping_csv('char_map.csv')

# Load character mapping
char_to_vector = load_char_mapping('char_map.csv')

# Visualize the mapping
visualize_char_mapping(char_to_vector)

# Convert a sentence to vector sequence
sentence = "Hello, world!"
sequence = sentence_to_vector_sequence(sentence, char_to_vector)
print(sequence.shape)  # (20, 2)

# Visualize a sentence
visualize_sentence(sentence, char_to_vector)

# Process a batch of sentences
sentences = ["Hello, world!", "This is a test", "Machine learning"]
batch = batch_to_vector_sequences(sentences, char_to_vector)
print(batch.shape)  # (3, 20, 2)

# Save training data
save_training_data(sentences, char_to_vector, 'training_data.npy')
"""