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

conn = sqlite3.connect("db_numprop-4_nestlim-100.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM data")
all_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
formula_idx = column_names.index("formula")
category_idx = column_names.index("category")
import sys
sys.setrecursionlimit(10000)



# ---------- Tree Node Definition with Hashing ----------
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

# ---------- Conversion Helper ----------
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

# Optimize the combine function which is a major bottleneck
@profile
def combine(l1, l2):
    if not l1 or not l2: 
        return []
    
    res = []
    for d1 in l1:
        for d2 in l2:
            # Fast incompatibility check
            compatible = True
            for k, v in d2.items():
                if k in d1 and d1[k] != v:
                    compatible = False
                    break
            
            if compatible:
                # Create new dict only when needed
                m = d1.copy()
                m.update(d2)
                res.append(m)
    return res
@lru_cache(maxsize=None)
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
    Uses sparse-to-dense vectorization to speed up distance computations.
    """
    assignments = tuple(input_row)
    key = (tree_candidate, correct, assignments)

    try:
        candidate_list, non_term_count = COMBINATION_CACHE[key]
    except KeyError:
        candidate_list, non_term_count = find_allowable_combinations(
            tree_candidate, correct, assignments
        )
        COMBINATION_CACHE[key] = (candidate_list, non_term_count)

    if not candidate_list or non_term_count == 0:
        return None, 0

    # Prepare prediction tensor [1, D]
    pred = nn_first_prediction.view(1, -1).to(device)
    N = len(candidate_list)
    D = non_term_count

    # Sanity check prediction length matches non_term_count
    if pred.size(1) != D:
        # If mismatch, pad or truncate prediction
        if pred.size(1) < D:
            pad = torch.zeros((1, D - pred.size(1)), device=device)
            pred = torch.cat([pred, pad], dim=1)
        else:
            pred = pred[:, :D]

    # Collect sparse indices and values for all candidates
    rows, cols, vals = [], [], []
    for i, cand in enumerate(candidate_list):
        for var_key, val in cand.items(): # var_key could be 'p', 'q', or an int
            if isinstance(var_key, int): # Check if it's one of our Z variables (now an int)
                idx = var_key
                # No need for startswith, slicing, or int(var_id[2:])
                if 0 <= idx < D:
                    rows.append(i)
                    cols.append(idx)
                    vals.append(val)

    # Build sparse tensor and convert to dense [N, D]
    if vals:
        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        values = torch.tensor(vals, dtype=torch.float, device=device)
        sparse = torch.sparse_coo_tensor(indices, values, (N, D), device=device)
        vectors = sparse.to_dense()
    else:
        vectors = torch.zeros((N, D), dtype=torch.float, device=device)

    # Compute L2 distances in one vectorized op
    dists = torch.norm(vectors - pred, dim=1)

    # Select best candidate
    best_idx = torch.argmin(dists).item()
    best_vec = vectors[best_idx]

    # Final sanity check: ensure best_vec length matches D
    if best_vec.numel() != D:
        # fallback to zero vector to avoid length mismatch
        return torch.zeros(D, device=device), D

    return best_vec, D



@profile
def train_on_truth_table(truth_table, bitlist, tree_candidate,
                         max_epochs=1000, lr=0.01, patience=10):
    """
    Trains a fresh Net(input_size, global_D) on the entire truth_table.
    global_D is the maximum non-terminal count over all rows.
    Returns epochs to converge, None if unsolvable, or inf if no convergence.
    """

    X = torch.from_numpy(truth_table.astype(np.float32)).to(device)
    N, input_size = X.size()

    # 1) Pre-scan to find each row's Dᵢ (use dummy pred of length 0)
    dummy_pred = torch.zeros(0, device=device)
    Ds = []
    for i in range(N):
        _, Di = compute_target(bitlist[i], tuple(truth_table[i]), dummy_pred, tree_candidate)
        if Di is None:
            return None # Unsolvable
        Ds.append(Di)

    if not Ds:
        return 0

    global_D = max(Ds) if Ds else 0
    if global_D == 0:
        return 0

    # 2) (Re)build the model with the correct output size
    # --- FIX APPLIED HERE ---
    # Directly instantiate the Net class
    model = Net(input_size, global_D).to(device)
    # -----------------------
    model.apply(init_weights)

    # 3) Prepare optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=3, verbose=False, min_lr=1e-5
    )

    # 4) Compute initial_preds (used for warm-start in target distance)
    with torch.no_grad():
        initial_preds = model(X) # Shape (N, global_D)

    # 5) Build padded target matrix Y (Optimized: Pre-allocate and fill)
    Y = torch.zeros(N, global_D, device=device)
    valid_indices_for_Y = [] # Keep track in case we need filtering later

    for i in range(N):
        pred_i = initial_preds[i] # Shape (global_D)
        tgt_i, _ = compute_target(bitlist[i], tuple(truth_table[i]), pred_i, tree_candidate)

        if tgt_i is None:
            return None # Unsolvable if any target cannot be computed

        numel = tgt_i.numel()
        if numel == 0 and global_D > 0:
             pass
        elif numel < global_D:
            Y[i, :numel] = tgt_i
        elif numel > global_D:
            Y[i, :] = tgt_i[:global_D]
        else: # numel == global_D
            Y[i, :] = tgt_i
        valid_indices_for_Y.append(i) # Track valid rows

    # Ensure we actually have data (especially if filtering were added)
    if len(valid_indices_for_Y) == 0 :
        return 0 # Or handle appropriately

    # Note: If you ever implement filtering for None targets, you'd need:
    # if len(valid_indices_for_Y) < N:
    #     X = X[valid_indices_for_Y]
    #     Y = Y[valid_indices_for_Y]
    #     N = X.shape[0] # Update N

    rounded_Y = round_prediction(Y)

    # 6) Standard training loop
    best_loss = float('inf')
    patience_ctr = 0
    check_every = 5
    loss_thresh = 1e-5

    for epoch in range(1, max_epochs + 1):
        model.train() # Ensure model is in training mode
        P = model(X) # Predictions, Shape (N, global_D)
        loss = F.binary_cross_entropy(P, Y)

        rounded_P_current_epoch = None
        check_condition = (epoch % check_every == 0) or (loss.item() < 0.001) # Check loss value, not tensor

        if check_condition:
            model.eval() # Set to eval mode for prediction rounding consistency
            with torch.no_grad():
                rounded_P_current_epoch = round_prediction(P)
            model.train() # Set back to train mode
            if torch.equal(rounded_P_current_epoch, rounded_Y):
                return epoch # Converged

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % 2 == 0:
            scheduler.step(loss.item())

        current_loss_val = loss.item()
        if current_loss_val < best_loss:
            best_loss = current_loss_val
            patience_ctr = 0
        else:
            patience_ctr += 1

        if patience_ctr >= patience and epoch > 50:
            # Check one last time before early stopping
            if rounded_P_current_epoch is None:
                model.eval()
                with torch.no_grad():
                    rounded_P_current_epoch = round_prediction(P)
                model.train()
            if torch.equal(rounded_P_current_epoch, rounded_Y):
                 return epoch
            return float('inf') # Did not converge within patience

        if current_loss_val < loss_thresh:
            if rounded_P_current_epoch is None:
                model.eval()
                with torch.no_grad():
                     rounded_P_current_epoch = round_prediction(P)
                model.train()
            if torch.equal(rounded_P_current_epoch, rounded_Y):
                 return epoch # Converged

    return float('inf') # Max epochs reached without convergence

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
    
    # Pre-compute truth table once
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
    row_indices = random.sample(range(len(all_rows)), 20)
    print(f"Testing with {len(row_indices)} random rows")
    
    for row_index in row_indices:
        print(f"Processing row {row_index}...")
        row = all_rows[row_index]
        final_formula = row[formula_idx]
        bitlist = binary_to_bitlist(row[category_idx], len(truth_table))
        first_correct = bitlist[0]
        first_assign = tuple(truth_table[0])


        current = None
        found = False    
        
        for depth in range(100):
            print(f"  Depth {depth}")
            current_options = run_derivation_for_row(row_index, row, column_names, current)
            result = evaluate_candidate_options(current_options, truth_table, final_formula, bitlist)
        
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