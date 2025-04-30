import sqlite3
import random
import networkx as nx
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

# ───────────────────────────────────────────────────────────────
# 1) GLOBAL CACHE for find_allowable_combinations results
# ───────────────────────────────────────────────────────────────
COMBINATION_CACHE = {}
@profile
def compute_target(correct, input_row, nn_first_prediction, tree_candidate):
    """
    Returns (best_target_tensor, non_terminal_count) or (None, 0).
    Caches the candidate list + count so we don't recompute it every call.
    """
    assignments = tuple(input_row)
    key = (id(tree_candidate), assignments)

    # fetch or compute the list of fillings + count
    if key not in COMBINATION_CACHE:
        candidate_list, non_term_count = find_allowable_combinations(
            tree_candidate, correct, assignments
        )
        COMBINATION_CACHE[key] = (candidate_list, non_term_count)
    else:
        candidate_list, non_term_count = COMBINATION_CACHE[key]

    if not candidate_list or non_term_count == 0:
        return None, 0

    # build a tensor for each candidate (shape: [non_term_count])
    vectors = []
    for cand in candidate_list:
        vec = torch.tensor(
            [cand.get(f"Z_{i}", 0) for i in range(non_term_count)],
            dtype=torch.float,
            device=device
        )
        vectors.append(vec)

    tensor_stack = torch.stack(vectors)               # [#candidates, non_term_count]
    # pick the one closest to the network’s prediction
    pred = nn_first_prediction.view(1, -1).to(device) # [1, non_term_count]
    dists = torch.norm(tensor_stack - pred, dim=1)    # [#candidates]
    best = torch.argmin(dists).item()

    return vectors[best], non_term_count


# ───────────────────────────────────────────────────────────────
# 2) BATCHED TRAINING FUNCTION
# ───────────────────────────────────────────────────────────────
@profile
def train_on_truth_table(nn_model, truth_table, bitlist, tree_candidate,
                         max_epochs=5000, lr=1e-2):
    """
    Trains nn_model on the entire truth_table as a single batch.
    Returns:
      - int  : epochs to converge
      - None : if some row is unsolvable
      - inf  : if it doesn't converge within max_epochs
    """
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=lr)
    X = torch.from_numpy(truth_table.astype(np.float32)).to(device)  # [N, in_size]
    N = X.size(0)

    # ——— STEP 1: compute ALL targets up front, and grab output_size ———
    targets = []
    output_size = None
    with torch.no_grad():
        for i in range(N):
            x_row = X[i:i+1]                        # [1, in_size]
            pred = nn_model(x_row).squeeze(0)       # [out_size]
            tgt, nt_count = compute_target(
                bitlist[i], tuple(truth_table[i]), pred, tree_candidate
            )
            if tgt is None:
                return None  # unsolvable
            # set output_size once, from the first row
            if output_size is None:
                output_size = nt_count
            # enforce consistency
            if tgt.numel() != output_size:
                raise ValueError(
                    f"Row {i} target length {tgt.numel()} != expected {output_size}"
                )
            targets.append(tgt)

    # stack into [N, output_size]
    Y = torch.stack(targets)
    assert Y.shape == (N, output_size)

    # ——— STEP 2: batch‐train until convergence or max_epochs ———
    for epoch in range(1, max_epochs+1):
        nn_model.train()
        P = nn_model(X)  # [N, output_size]

        # check if rounding already matches
        if torch.equal(round_prediction(P), round_prediction(Y)):
            return epoch

        loss = F.binary_cross_entropy(P, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return float('inf')


@profile
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
        for depth in range(30):
            COMBINATION_CACHE.clear()
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