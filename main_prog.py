import sqlite3
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time


device = torch.device("cpu")
print(f"Using device: {device}")
from helpers import (find_allowable_combinations, round_prediction,
                     binary_to_bitlist, tree_to_formula, run_derivation_for_row, check_tree_matches, percent_longer)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.ln1 = nn.Linear(input_size, 16)
        self.ln2 = nn.Linear(16, 16)
        self.ln3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = 2 * x - 1
        x = x.to(device)
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return torch.sigmoid(x)


MODEL_CACHE = {}
def get_model(input_size, output_size):
    if output_size not in MODEL_CACHE:
        model = Net(input_size, output_size).to(device)
        MODEL_CACHE[output_size] = model
    model = MODEL_CACHE[output_size]
    model.apply(init_weights)  
    return model

# Compute the target vector for a given row and tree candidate
def compute_target(correct, input_row, nn_first_prediction, tree_candidate):
    assignments = tuple(input_row)
    candidate_list, non_terminal_count = find_allowable_combinations(tree_candidate, correct, assignments)

    if not candidate_list or non_terminal_count == 0:
        return None, 0

    candidate_vectors = [
        torch.tensor(
            [candidate.get(f"Z_{i}", 0) for i in range(non_terminal_count)],
            dtype=torch.float,
            device=device
        )
        for candidate in candidate_list
    ]

    if not candidate_vectors:
        return None, 0

    candidate_tensor = torch.stack(candidate_vectors)
    nn_pred = nn_first_prediction.view(1, -1).to(device)
    distances = torch.norm(candidate_tensor - nn_pred, dim=1)
    best_idx = torch.argmin(distances).item()
    return candidate_vectors[best_idx], non_terminal_count

# Improved train_on_truth_table function
def train_on_truth_table(nn_model, truth_table, bitlist, tree_candidate, max_iterations=5000):
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.01)
    truth_tensor = torch.from_numpy(truth_table.astype(np.float32)).to(device)
    
    # First, get the output size and check if we can solve this candidate
    first_row = truth_tensor[0].unsqueeze(0)
    with torch.no_grad():
        dummy_pred = nn_model(first_row).squeeze(0)
    
    first_target, output_size = compute_target(bitlist[0], tuple(truth_table[0]), dummy_pred, tree_candidate)
    if first_target is None:
        return None
    
    # Check if all rows can be solved with this tree candidate
    all_solvable = True
    for i in range(len(truth_table)):
        with torch.no_grad():
            row_pred = nn_model(truth_tensor[i].unsqueeze(0)).squeeze(0)
        row_target, _ = compute_target(bitlist[i], tuple(truth_table[i]), row_pred, tree_candidate)
        if row_target is None:
            all_solvable = False
            break
    
    if not all_solvable:
        return None
        
    # Training loop - process one row at a time
    iterations_per_row = []
    max_iter = 1000
    
    for i, row in enumerate(truth_table):
        row_tensor = torch.from_numpy(row.astype(np.float32)).unsqueeze(0).to(device)
        correct = bitlist[i]
        
        training_iter = 0
        while training_iter < max_iter:
            # Forward pass
            pred = nn_model(row_tensor)
            pred_flat = pred.squeeze(0)
            
            # Get target for this prediction
            target, _ = compute_target(correct, tuple(row), pred_flat, tree_candidate)
            if target is None:
                return None
                
            # Check if we've converged
            pred_binary = round_prediction(pred_flat)
            if torch.equal(pred_binary, target):
                break
                
            # Compute loss and update
            loss = F.binary_cross_entropy(pred, target.unsqueeze(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            training_iter += 1
            
        if training_iter == max_iter:
            return float('inf')  # This candidate is too difficult to train
            
        iterations_per_row.append(training_iter)
    
    # Return the total number of iterations
    return sum(iterations_per_row)

def evaluate_candidate_options(current_options, truth_table, final_formula, bitlist):
    candidate_iterations = {}

    for idx, tree_cand in current_options.items():
        cand_str = tree_to_formula(tree_cand)

        # 1) Exact match to your known minimal formula?
        if cand_str == final_formula:
            return cand_str

        # 2) Fully instantiated (no Z_i’s) → test on every row
        if 'Z' not in cand_str:
            all_correct = True
            for i, row in enumerate(truth_table):
                if not check_tree_matches(tree_cand, bitlist[i], tuple(row)):
                    all_correct = False
                    break

            if all_correct:
                return cand_str
            else:
                continue

        # 3) Otherwise has nonterminals → try fillings via NN
        first_assignment = tuple(truth_table[0])
        candidate_list, non_terminal_count = find_allowable_combinations(
            tree_cand, bitlist[0], first_assignment
        )
        if non_terminal_count == 0 or not candidate_list:
            continue

        nn_model = get_model(truth_table.shape[1], non_terminal_count)
        iters = train_on_truth_table(nn_model, truth_table, bitlist, tree_cand)
        if iters is None or iters == float('inf'):
            continue

        candidate_iterations[idx] = iters

    return candidate_iterations

if __name__ == "__main__":
    conn = sqlite3.connect("db_numprop-4_nestlim-100.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM data")
    all_rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]
    formula_idx = column_names.index("formula")
    category_idx = column_names.index("category")
    start_time = time.time()
    
    truth_table = np.array([
        [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0],
        [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],
        [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0],
        [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]
    ])

    found_count = 0
    minimal_count = 0
    percent_diffs = []           # will collect each row’s %‐overlength
    found_formulas = []

    # Process all rows with progress tracking
    for row_index in range(100):
        print(f"Processing row {row_index}...")
        row = all_rows[row_index]
        final_formula = row[formula_idx]
        bitlist = binary_to_bitlist(row[category_idx], len(truth_table))

        current = None
        for depth in range(20):
            print(f"  Depth {depth}")
            current_options = run_derivation_for_row(row_index, row, column_names, current)
            result = evaluate_candidate_options(current_options, truth_table, final_formula, bitlist)

            if isinstance(result, str):
                found_count += 1
                found_formula = result
                minimal = final_formula

                # is it *exactly* the minimal?
                if found_formula == minimal:
                    minimal_count += 1
                    pct = 0.0
                else:
                    pct = percent_longer(found_formula, minimal)

                percent_diffs.append(pct)
                found_formulas.append((row_index, found_formula, pct))
                print(f"  Found formula: {found_formula}  ({pct:.1f}% longer)")
                break
            elif result:  # Check if result is not empty
                best_key = min(result, key=result.get)
                current = current_options[best_key]
                print(f"  Best candidate: {tree_to_formula(current)} (iterations: {result[best_key]})")
            else:
                # No viable candidates found
                print("  No viable candidates found")
                break


    print(f"\n=== Summary ===")
    print(f"  Rows with *any* formula found : {found_count} out of {1000}")
    print(f"  Of those, exact minimals     : {minimal_count}")
    if percent_diffs:
        avg_pct = sum(percent_diffs) / len(percent_diffs)
        print(f"  Average %‐overlength: {avg_pct:.1f}%")

    for idx, formula, pct in found_formulas:
        sign = '+' if pct >= 0 else ''
        print(f" • row {idx}: {formula!r} → {sign}{pct:.1f}%")

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds.")