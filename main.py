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

conn = sqlite3.connect("db_numprop-4_nestlim-100.db")
cursor = conn.cursor()

cursor.execute("SELECT * FROM data")
all_rows = cursor.fetchall()
column_names = [desc[0] for desc in cursor.description]
formula_idx = column_names.index("formula")
category_idx = column_names.index("category")

# ---------- Tree Node Definition with Hashing ----------
class TreeNode:
    __slots__ = ('op', 'left', 'right')
    def __init__(self, op, left=None, right=None):
        self.op = op
        self.left = left
        self.right = right
    def __repr__(self):
        if self.op in ('p','q','r','s','Z'):
            return self.op
        if self.op=='N':
            return f"(N {self.left})"
        return f"({self.op} {self.left} {self.right})"
    def __eq__(self, other):
        return isinstance(other, TreeNode) and (self.op,self.left,self.right)==(other.op,other.left,other.right)
    def __hash__(self):
        return hash((self.op,self.left,self.right))

# ---------- Conversion Helper ----------
# Cache frequently converted trees
TREENODE_CACHE = {}
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
def _find_combinations(node, correct, assignments, x_counter):
    p, q, r, s = assignments
    vm = {'p': p, 'q': q, 'r': r, 's': s}
    
    op = node.op
    
    # Fast path for terminals
    if op in vm:
        return ([{}] if vm[op] == correct else []), x_counter
    if op == 'Z':
        name = f"Z_{x_counter}"
        return ([{name: correct}]), x_counter + 1
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

device = torch.device("cuda")
print(f"Using device: {device}")
from helpers import (round_prediction, binary_to_bitlist, 
                     tree_to_formula, run_derivation_for_row, 
                     check_tree_matches, percent_longer)

def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# Optimized network - reduced size for faster training
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        # Simplified architecture
        self.ln1 = nn.Linear(input_size, 12)  # Reduced from 16
        self.ln2 = nn.Linear(12, output_size)  # Removed third layer

    def forward(self, x):
        x = 2 * x - 1
        x = F.relu(self.ln1(x))
        x = self.ln2(x)
        return torch.sigmoid(x)

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
    """
    assignments = tuple(input_row)
    
    # Create a hashable key for the cache - use tuple representation of tree instead of id
    # This is more reliable if trees are recreated but structurally identical
    key = (tuple(str(tree_candidate).encode()), correct, assignments)
    
    # fetch or compute the list of fillings + count
    if key not in COMBINATION_CACHE:
        candidate_list, non_term_count = find_allowable_combinations(
            tree_candidate, correct, assignments
        )
        COMBINATION_CACHE[key] = (candidate_list, non_term_count)
    else:
        candidate_list, non_term_count = COMBINATION_CACHE[key]
    
    # Early return if no valid candidates or non-terminals
    if not candidate_list or non_term_count == 0:
        return None, 0
    
    # Optimize tensor creation for large candidate lists
    if len(candidate_list) > 1000:
        # Process in batches to reduce memory pressure
        batch_size = 1000
        min_dist = float('inf')
        best_vector = None
        
        for i in range(0, len(candidate_list), batch_size):
            batch = candidate_list[i:i+batch_size]
            
            # Create tensor in one operation for efficiency
            batch_data = torch.zeros((len(batch), non_term_count), dtype=torch.float, device=device)
            
            # OPTIMIZED: Use vectorized operations instead of nested loops
            # Pre-process the batch to build coordinate format for sparse tensor
            indices_j = []
            indices_idx = []
            values = []
            
            for j, cand in enumerate(batch):
                for var_id, val in cand.items():
                    if var_id.startswith("Z_"):
                        idx = int(var_id[2:])
                        if idx < non_term_count:
                            indices_j.append(j)
                            indices_idx.append(idx)
                            values.append(val)
            
            # Use vectorized indexing to set all values at once
            if values:  # Only if we have values to set
                batch_data[indices_j, indices_idx] = torch.tensor(values, dtype=torch.float, device=device)
            
            # Calculate distances in one operation
            pred = nn_first_prediction.view(1, -1).to(device)
            batch_dists = torch.norm(batch_data - pred, dim=1)
            
            # Update best if found
            batch_min_idx = torch.argmin(batch_dists).item()
            batch_min_dist = batch_dists[batch_min_idx].item()
            
            if batch_min_dist < min_dist:
                min_dist = batch_min_dist
                best_vector = batch_data[batch_min_idx]
        
        return best_vector, non_term_count
    
    else:
        # For smaller lists, use a more efficient approach
        # Create list of dictionaries first
        vector_values = []
        for cand in candidate_list:
            # Extract only the values that match the pattern and are in range
            vec_dict = {int(k[2:]): v for k, v in cand.items() 
                       if k.startswith("Z_") and int(k[2:]) < non_term_count}
            vector_values.append(vec_dict)
        
        # Create tensor all at once
        vectors = torch.zeros((len(candidate_list), non_term_count), dtype=torch.float, device=device)
        
        # Populate tensor using vectorized operations where possible
        for i, vec_dict in enumerate(vector_values):
            if vec_dict:  # Only if we have values to set
                indices = list(vec_dict.keys())
                values = list(vec_dict.values())
                indices_tensor = torch.tensor(indices, dtype=torch.long, device=device)
                values_tensor = torch.tensor(values, dtype=torch.float, device=device)
                vectors[i].index_copy_(0, indices_tensor, values_tensor)
        
        # Pick the one closest to the network's prediction
        pred = nn_first_prediction.view(1, -1).to(device)  # [1, non_term_count]
        dists = torch.norm(vectors - pred, dim=1)          # [#candidates]
        best = torch.argmin(dists).item()
        
        return vectors[best], non_term_count

@profile
def train_on_truth_table(nn_model, truth_table, bitlist, tree_candidate, 
                           max_epochs=1000, lr=0.01, patience=10):
    """
    Trains nn_model on the entire truth_table as a single batch.
    Returns:
      - int  : epochs to converge
      - None : if some row is unsolvable
      - inf  : if it doesn't converge within max_epochs
    """
    # Switch to a faster optimizer with better parameters
    optimizer = torch.optim.AdamW(nn_model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Use learning rate scheduler to speed up convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=False
    )
    
    # Convert data once and ensure it's on the right device
    X = torch.from_numpy(truth_table.astype(np.float32)).to(device)
    N = X.size(0)
    
    # Pre-compute all targets up front to avoid repeated computation
    targets = []
    output_size = None
    
    # Use a single forward pass to get all predictions
    with torch.no_grad():
        initial_preds = nn_model(X)
        
        for i in range(N):
            pred = initial_preds[i]
            tgt, nt_count = compute_target(
                bitlist[i], tuple(truth_table[i]), pred, tree_candidate
            )
            
            if tgt is None:
                return None  # unsolvable
                
            # Set output_size once
            if output_size is None:
                output_size = nt_count
                
            # Enforce consistency
            if tgt.numel() != output_size:
                raise ValueError(
                    f"Row {i} target length {tgt.numel()} != expected {output_size}"
                )
                
            targets.append(tgt)
    
    # Stack into single tensor
    Y = torch.stack(targets)
    
    # Cache rounded Y for faster comparisons
    rounded_Y = round_prediction(Y)
    
    # Training loop with improved early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(1, max_epochs + 1):
        # Standard forward pass (autocast not safe with binary_cross_entropy)
        P = nn_model(X)
        loss = F.binary_cross_entropy(P, Y)
        
        # Early stopping check - only do rounding when necessary
        # and use cached tensors where possible
        if epoch % 5 == 0 or loss < 0.001:  # Only check periodically or when loss is very low
            rounded_P = round_prediction(P)
            if torch.equal(rounded_P, rounded_Y):
                return epoch
        
        # More efficient gradient reset
        optimizer.zero_grad(set_to_none=True)
        
        # Standard backward and optimization
        loss.backward()
        optimizer.step()
        
        # Update learning rate based on loss
        scheduler.step(loss.item())
        
        # Implement better early stopping
        if loss < best_loss:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience and epoch > 50:
            # Check if we've actually converged before stopping
            rounded_P = round_prediction(P)
            if torch.equal(rounded_P, rounded_Y):
                return epoch
                
        # Fast exit on very low loss
        if loss < 1e-5:
            rounded_P = round_prediction(P)
            if torch.equal(rounded_P, rounded_Y):
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
        iters = train_on_truth_table(nn_model, truth_table, bitlist, tree_cand)
        
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
    row_indices = random.sample(range(len(all_rows)), 3)
    print(f"Testing with {len(row_indices)} random rows")
    
    for row_index in row_indices:
        print(f"Processing row {row_index}...")
        row = all_rows[row_index]
        final_formula = row[formula_idx]
        bitlist = binary_to_bitlist(row[category_idx], len(truth_table))

        current = None
        found = False    
        
        for depth in range(50):  # Reduced from 50 to 10 for faster execution
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
            elif result:
                best_key = min(result, key=result.get)
                current = current_options[best_key]
                print(f"  Best candidate: {tree_to_formula(current)} (iterations: {result[best_key]})")
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