import sqlite3
import random
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from functools import lru_cache
from collections import defaultdict
device = torch.device("cuda")
print(f"Using device: {device}")
from helpers import (round_prediction, binary_to_bitlist, 
                     tree_to_formula, run_derivation_for_row, 
                     check_tree_matches, percent_longer)

conn = sqlite3.connect("db_sampled_10percent_stratified2.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM data")
all_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
formula_idx = column_names.index("formula")
category_idx = column_names.index("category")
import sys
sys.setrecursionlimit(10000)




class TreeNode:
    def __init__(self, op, left=None, right=None):
        self.op = op
        self.left = left
        self.right = right
        # Precompute hash if tree structure is immutable after creation
        self._hash = None

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.op == other.op and self.left == other.left and self.right == other.right

    def __hash__(self):
        if self._hash is None: # Compute and cache hash if not already done
            # Ensure children are also hashable (None is fine)
            self._hash = hash((self.op, self.left, self.right))
        return self._hash
    def __repr__(self):
        if self.left is None and self.right is None:
            return f"TreeNode({self.op!r})"
        return f"TreeNode({self.op!r}, left={self.left!r}, right={self.right!r})"

# Cache frequently converted trees
TREENODE_CACHE = {}
@profile
def tuple_to_treenode(tree):
    # Add caching to reduce conversion time
    if isinstance(tree, TreeNode): 
        return tree
    if tree in TREENODE_CACHE:
        return TREENODE_CACHE[tree]
    
    if isinstance(tree, str): 
        result = TreeNode(tree)
    else:
        op, *args = tree
        if op == 'N': 
            result = TreeNode('N', left=tuple_to_treenode(args[0]))
        else:
            left = tuple_to_treenode(args[0])
            right = tuple_to_treenode(args[1]) if len(args) > 1 else None
            result = TreeNode(op, left=left, right=right)
    
    if not isinstance(tree, TreeNode) and not isinstance(tree, str):
        TREENODE_CACHE[tree] = result
    return result

# Precomputed truth tables for binary ops
TRUTH_TABLES = {
    'A': {  
        1: [(1, 1)], 
        0: [(0, 0), (0, 1), (1, 0)] 
    },
    'O': {  
        1: [(1, 1), (1, 0), (0, 1)], 
        0: [(0, 0)]  
    },
    'C': {  
        1: [(0, 0), (0, 1), (1, 1)],  
        0: [(1, 0)]  
    },
    'NC': {  
        1: [(1, 0)],
        0: [(0, 0), (0, 1), (1, 1)] 
    },
    'B': { 
        1: [(1, 1), (0, 0)], 
        0: [(1, 0), (0, 1)]
    },
    'X': {  
        1: [(1, 0), (0, 1)],
        0: [(1, 1), (0, 0)]
    },
    'NA': {  
        1: [(0, 0), (0, 1), (1, 0)],
        0: [(1, 1)]
    },
    'NOR': {
        1: [(0, 0)], 
        0: [(1, 1), (1, 0), (0, 1)] 
    }
}

@profile
def combine(l1, l2):
    """Safely merge two lists of partial assignment dicts, filtering incompatible combinations."""
    if not l1 or not l2:
        return []

    # Ensure smaller list is inner loop
    if len(l1) < len(l2):
        shorter, longer = l1, l2
        swap = False
    else:
        shorter, longer = l2, l1
        swap = True

    res = []
    for d_small in shorter:
        if not isinstance(d_small, dict):
            continue
        for d_large in longer:
            if not isinstance(d_large, dict):
                continue
            d1, d2 = (d_large, d_small) if swap else (d_small, d_large)
            try:
                if all(d1.get(k, v) == v for k, v in d2.items()):
                    merged = d1.copy()
                    merged.update(d2)
                    res.append(merged)
            except Exception as e:
                # Gracefully skip bad merges
                continue
    return res


@lru_cache(maxsize=20000)
@profile
def _find_combinations(node, correct, assignments, x_counter):
    p, q, r, s = assignments
    vm = {'p': p, 'q': q, 'r': r, 's': s}
    
    op = node.op
    
    # Fast path for terminals
    if op in vm:
        return ([{}] if vm[op] == correct else []), x_counter
    if op == 'Z':
        return ([{x_counter: correct}]), x_counter + 1 # New, x_counter is an int
    if op == 'N':
        return _find_combinations(node.left, 1 - correct, assignments, x_counter)
    
    # Binary operations
    left, right = node.left, node.right
    pairs = TRUTH_TABLES[op][correct]
    
    # Pre-allocate for potentially large result
    all_res = []
    max_c = x_counter
    
    # Process in batches to reduce memory pressure
    for lv, rv in pairs:
        left_list, c1 = _find_combinations(left, lv, assignments, x_counter)
        max_c = max(max_c, c1)
        
        if not left_list: 
            continue
            
        right_list, c2 = _find_combinations(right, rv, assignments, c1)
        max_c = max(max_c, c2)
        
        if not right_list: 
            continue
            
        # This is the most expensive operation - optimize the combine
        batch_results = combine(left_list, right_list)
        all_res.extend(batch_results)
    
    return all_res, max_c

# Public API with result caching
COMBINATION_CACHE = {}

@lru_cache(maxsize=100000)
def find_allowable_combinations(tree, correct, assignments, x_counter=0):
    node = tuple_to_treenode(tree)
    return _find_combinations(node, correct, tuple(assignments), x_counter)



class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.ln1 = nn.Linear(input_size, 16)
        self.ln2 = nn.Linear(16, 16)
        self.ln3 = nn.Linear(16, output_size)

    def forward(self, x):
        x = 2 * x - 1
        x = F.relu(self.ln1(x))
        x = F.relu(self.ln2(x))
        x = self.ln3(x)
        return torch.sigmoid(x)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

MODEL_CACHE = {}
def get_model(input_size, output_size):
    if output_size not in MODEL_CACHE:
        model = Net(input_size, output_size).to(device)
        MODEL_CACHE[output_size] = model
    model = MODEL_CACHE[output_size]
    model.apply(init_weights)  
    return model


@profile
def compute_target(correct, input_row, nn_first_prediction, tree_candidate):
    """
    Returns (best_target_tensor, non_terminal_count) or (None, 0).
    Caches the candidate list + count so we don't recompute it every call.
    Uses precomputed non-terminal counts to speed up operations.
    """
    assignments = tuple(input_row)
    #cache_key = (tree_candidate, correct, assignments)

    # Retrieve or compute allowable combinations and non-terminal count
    #try:
       # candidate_list, non_term_count = COMBINATION_CACHE[cache_key]
    #except KeyError:
    candidate_list, non_term_count = find_allowable_combinations(
            tree_candidate, correct, assignments
        )
        #COMBINATION_CACHE[cache_key] = (candidate_list, non_term_count)

    D = non_term_count
    # Quick exit if no combinations or no non-terminals
    if not candidate_list or D == 0:
        return None, 0

    # Prepare the prediction vector [1, D]
    pred = nn_first_prediction.view(1, -1).to(device)
    # Pad or truncate to match D
    if pred.size(1) != D:
        pad_width = max(0, D - pred.size(1))
        pred = torch.nn.functional.pad(pred, (0, pad_width), mode='constant', value=0)
        pred = pred[:, :D]

    # Build sparse indices for Z-variables (integer keys < D)
    rows, cols, vals = [], [], []
    for i, cand in enumerate(candidate_list):
        for var_key, val in cand.items():
            if isinstance(var_key, int) and 0 <= var_key < D:
                rows.append(i)
                cols.append(var_key)
                vals.append(val)

    # Construct dense vectors from sparse representation
    if vals:
        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        values = torch.tensor(vals, dtype=torch.float, device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (len(candidate_list), D), device=device)
        vectors = sparse.to_dense()
    else:
        vectors = torch.zeros((len(candidate_list), D), device=device)

    # Compute distances and select best
    dists = torch.norm(vectors - pred, dim=1)
    best_idx = torch.argmin(dists).item()
    best_vec = vectors[best_idx]

    # Ensure best_vec has length D
    if best_vec.numel() != D:
        return torch.zeros(D, device=device), D

    return best_vec, D








@profile
def train_on_truth_table(truth_table, bitlist, tree_candidate,
                             max_epochs=1000, lr=0.01, patience=10):
    """
    Trains a fresh Net(input_size, global_D) on the entire truth_table.
    Returns epochs to converge, None if unsolvable, or inf if no convergence.
    Optimized to call compute_target only once per row.
    """
    # Convert truth table to tensor
    X = torch.from_numpy(truth_table.astype(np.float32)).to(device)
    N, input_size = X.size()

    # First pass: compute targets and non-terminal counts, cache results
    target_list = []  # list of (tgt_tensor, D)
    Ds = []
    dummy_pred = torch.zeros(0, device=device)
    for i in range(N):
        tgt_i, Di = compute_target(bitlist[i], tuple(truth_table[i]), dummy_pred, tree_candidate)
        if tgt_i is None:
            return None  # Unsolvable row
        target_list.append((tgt_i, Di))
        Ds.append(Di)

    if not Ds:
        return 0
    global_D = max(Ds)
    if global_D == 0:
        return 0

    # Prepare model and optimizer
    model = Net(input_size, global_D).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3, verbose=False, min_lr=1e-5
    )

    # Build padded target matrix Y
    Y = torch.zeros(N, global_D, device=device)
    for i, (tgt_i, Di) in enumerate(target_list):
        # pad or truncate each target vector to global_D
        if Di < global_D:
            pad = torch.zeros(global_D - Di, device=device)
            Y[i] = torch.cat([tgt_i, pad])
        else:
            Y[i] = tgt_i[:global_D]

    # Initial prediction (unused for training but could be used for warm start)
    with torch.no_grad():
        _ = model(X)

    # Training loop
    best_loss = float('inf')
    patience_ctr = 0
    check_every = 5
    loss_thresh = 1e-5

    for epoch in range(1, max_epochs + 1):
        model.train()
        P = model(X)
        loss = F.binary_cross_entropy(P, Y)

        # Periodic checking for convergence
        rounded_correct = None
        if epoch % check_every == 0 or loss.item() < 0.001:
            model.eval()
            with torch.no_grad():
                rounded_correct = round_prediction(P)
            model.train()
            if torch.equal(rounded_correct, round_prediction(Y)):
                return epoch

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 2 == 0:
            scheduler.step(loss.item())

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience and epoch > 50:
                model.eval()
                with torch.no_grad():
                    rounded_correct = round_prediction(model(X))
                if torch.equal(rounded_correct, round_prediction(Y)):
                    return epoch
                return float('inf')

        if current_loss < loss_thresh:
            model.eval()
            with torch.no_grad():
                rounded_correct = round_prediction(model(X))
            if torch.equal(rounded_correct, round_prediction(Y)):
                return epoch

    return float('inf')

@profile
def evaluate_candidate_options(current_options, truth_table, final_formula, bitlist):
    candidate_iterations = {}
    
    # First check exact matches without NN training
    for idx, tree_cand in current_options.items():
        cand_str = tree_to_formula(tree_cand)
        
        # 1) Exact match to known minimal formula?
        if cand_str == final_formula:
            return cand_str
            
        # 2) Fully instantiated (no Z_i's) → test on every row
        if 'Z' not in cand_str:
            all_correct = True
            for i, row in enumerate(truth_table):
                if not check_tree_matches(tree_cand, bitlist[i], tuple(row)):
                    all_correct = False
                    break
                    
            if all_correct:
                return cand_str
    
    # Now try NN training for the candidates with nonterminals
    for idx, tree_cand in current_options.items():
        cand_str = tree_to_formula(tree_cand)
        
        # Skip already checked fully instantiated formulas
        if 'Z' not in cand_str:
            continue
            
        # Pre-check if candidate can work with first row
        first_assignment = tuple(truth_table[0])
        candidate_list, non_terminal_count = find_allowable_combinations(
            tree_cand, bitlist[0], first_assignment
        )
        
        if non_terminal_count == 0 or not candidate_list:
            continue
            
        # Train model with optimized function
        nn_model = get_model(truth_table.shape[1], non_terminal_count)
        iters = train_on_truth_table(truth_table, bitlist, tree_cand)
        
        if iters is None or iters == float('inf'):
            continue
            
        candidate_iterations[idx] = iters
    
    return candidate_iterations

if __name__ == "__main__":
    unfound_rows = []
    start_time = time.time()
    
    truth_table = np.array([
        [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 1], [1, 1, 0, 0],
        [1, 0, 1, 1], [1, 0, 1, 0], [1, 0, 0, 1], [1, 0, 0, 0],
        [0, 1, 1, 1], [0, 1, 1, 0], [0, 1, 0, 1], [0, 1, 0, 0],
        [0, 0, 1, 1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]
    ])

    found_count = 0
    minimal_count = 0
    percent_diffs = []           
    found_formulas = []
    # Sample fewer rows for faster execution during optimization 
    row_indices = random.sample(range(len(all_rows)), 5)
    print(f"Testing with {len(row_indices)} random rows")
    
    for row_index in range(300):
        print(f"Processing row {row_index}...")
        row = all_rows[row_index]
        final_formula = row[formula_idx]
        bitlist = binary_to_bitlist(row[category_idx], len(truth_table))
        first_correct = bitlist[0]
        first_assign = tuple(truth_table[0])


        current = None
        found = False    

        for depth in range(250):
            print(f"  Depth {depth}")
            current_options = run_derivation_for_row(row_index, row, column_names, current)
            derivation_start_time = time.time()
            result = evaluate_candidate_options(current_options, truth_table, final_formula, bitlist)
            derivation_duration = time.time() - derivation_start_time
            
            if derivation_duration > 1.25:
                print('Stop, timeout')
                break

            if isinstance(result, str):
                found = True
                found_count += 1
                found_formula = result
                minimal = final_formula

                if found_formula == minimal:
                    minimal_count += 1
                    pct = 0.0
                else:
                    pct = percent_longer(found_formula, minimal)

                percent_diffs.append(pct)
                found_formulas.append((row_index, found_formula, pct))
                print(f"  Found formula: {found_formula}  ({pct:.1f}% longer)")
                break
             # Case B: we have a dict of {option_index: iterations}
            elif result:
                # sort candidates by fewest NN-iterations first
                sorted_candidates = sorted(result.items(), key=lambda kv: kv[1])

                picked = False
                for opt_idx, iters in sorted_candidates:
                    cand_tree = current_options[opt_idx]

                    #check viability before accepting it
                    combs, _ = find_allowable_combinations(
                        cand_tree,
                        first_correct,                 
                        first_assign,      
                        x_counter=0                         
                    )
                    if combs:
                        # this candidate can actually instantiate for this row
                        current = cand_tree
                        print(f"  Best viable candidate: {tree_to_formula(current)} (iterations: {iters})")
                        picked = True
                        break

                if not picked:
                    print("  No *viable* candidates found at this depth—bailing out")
                    break

            # Case C: no candidates at all
            else:
                print("  No viable candidates found")
                break

        if not found:
            unfound_rows.append((row_index, final_formula))

    print(f"\n=== Summary ===")
    print(f"  Rows with *any* formula found : {found_count} out of {len(row_indices)}")
    print(f"  Of those, exact minimals     : {minimal_count}")
    if percent_diffs:
        avg_pct = sum(percent_diffs) / len(percent_diffs)
        print(f"  Average %‐overlength: {avg_pct:.1f}%")

    for idx, formula, pct in found_formulas:
        sign = '+' if pct >= 0 else ''
        print(f" • row {idx}: {formula!r} → {sign}{pct:.1f}%")

    # Summarize % overlength by minimal formula length
    length_groups = defaultdict(list)
    
    for (row_index, found_formula, pct) in found_formulas:
        minimal = all_rows[row_index][formula_idx]
        minimal_length = len(minimal)
        length_groups[minimal_length].append(pct)
    
    print("\n=== Overlength Summary by Minimal Formula Length ===")
    for length in sorted(length_groups):
        diffs = length_groups[length]
        avg_pct = sum(diffs) / len(diffs) if diffs else 0
        print(f"  Minimal length {length}: {len(diffs)} formulas, average %‐overlength = {avg_pct:.1f}%")

    if unfound_rows:
        print(f"\n=== Rows where no formula was found ({len(unfound_rows)}) ===")
    for idx, minimal_formula in unfound_rows:
        print(f" • row {idx}: minimal formula was {minimal_formula!r}")

    elapsed = time.time() - start_time
    print(f"Total execution time: {elapsed:.2f} seconds.")
    print(len(COMBINATION_CACHE))
