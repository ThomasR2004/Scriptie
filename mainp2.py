import sqlite3
import random
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import sys
import os
from functools import lru_cache
from collections import defaultdict
import concurrent.futures
import multiprocessing as mp
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from helpers import (round_prediction, binary_to_bitlist,
tree_to_formula, run_derivation_for_row,
check_tree_matches, percent_longer, find_allowable_combinations)




device = torch.device("cpu")
print(f"Using device: {device}", flush=True)



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
    cache_key = (input_size, output_size)
    if cache_key not in MODEL_CACHE:
        model = Net(input_size, output_size).to(device)
        MODEL_CACHE[cache_key] = model
    model = MODEL_CACHE[cache_key]
    model.apply(init_weights)
    return model

def compute_target(correct_val, input_row_tuple, nn_initial_prediction_tensor, tree_candidate_tuple_repr):
    assignments_tuple = tuple(input_row_tuple)

    candidate_assignment_list, num_Z_variables = find_allowable_combinations(
        tree_candidate_tuple_repr, correct_val, assignments_tuple
    )

    D = num_Z_variables

    if not candidate_assignment_list or D == 0:
        return None, D

    # Ensure prediction tensor shape matches D
    pred_tensor = nn_initial_prediction_tensor.view(1, -1).to(device)
    if pred_tensor.size(1) != D:
        pad_width = max(0, D - pred_tensor.size(1))
        pred_tensor = F.pad(pred_tensor, (0, pad_width), mode='constant', value=0)
        pred_tensor = pred_tensor[:, :D]

    # Convert candidate assignments to dense tensor format
    rows, cols, vals = [], [], []
    for i, cand_dict in enumerate(candidate_assignment_list):
        for var_key, val in cand_dict.items():
            if isinstance(var_key, int) and 0 <= var_key < D:
                rows.append(i)
                cols.append(var_key)
                vals.append(val)

    if vals:
        indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
        values = torch.tensor(vals, dtype=torch.float, device=device)
        num_candidates = len(candidate_assignment_list)
        sparse_candidate_vectors = torch.sparse_coo_tensor(indices, values, (num_candidates, D), device=device)
        dense_candidate_vectors = sparse_candidate_vectors.to_dense()
    else:
        dense_candidate_vectors = torch.zeros((len(candidate_assignment_list), D), device=device)

    # Find the candidate closest to the prediction
    distances = torch.norm(dense_candidate_vectors - pred_tensor, dim=1)
    best_idx = torch.argmin(distances).item()
    best_target_vector = dense_candidate_vectors[best_idx]

    return best_target_vector, D

def train_on_truth_table(truth_table_np, bitlist_target, tree_candidate_tuple_repr,
                             max_epochs=1000, lr=0.01, patience=10):
    X_tensor = torch.from_numpy(truth_table_np.astype(np.float32)).to(device)
    N_rows, input_feature_size = X_tensor.size()

    target_info_list = []
    D_values_for_rows = []

    dummy_nn_prediction = torch.empty(0, device=device)

    for i in range(N_rows):
        input_row_tuple = tuple(truth_table_np[i])
        target_vector_i, D_i = compute_target(bitlist_target[i], input_row_tuple,
                                              dummy_nn_prediction, tree_candidate_tuple_repr)
        if target_vector_i is None:
            return None
        target_info_list.append((target_vector_i, D_i))
        D_values_for_rows.append(D_i)

    if not D_values_for_rows:
        return 0

    global_D = max(D_values_for_rows) if D_values_for_rows else 0
    if global_D == 0:
        return 0

    model = get_model(input_feature_size, global_D)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.3, patience=max(1, patience // 3), min_lr=1e-6
    )

    Y_target_matrix = torch.zeros(N_rows, global_D, device=device)
    for i, (target_vector_i, D_i) in enumerate(target_info_list):
        if D_i > 0 and target_vector_i is not None: # Ensure target_vector_i is not None
            len_to_copy = min(D_i, global_D, target_vector_i.numel())
            Y_target_matrix[i, :len_to_copy] = target_vector_i[:len_to_copy]

    best_loss_val = float('inf')
    patience_counter = 0
    convergence_check_freq = 5
    loss_convergence_threshold = 1e-5

    for epoch in range(1, max_epochs + 1):
        model.train()
        predictions = model(X_tensor)
        loss = F.binary_cross_entropy(predictions, Y_target_matrix)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        current_loss_val = loss.item()
        scheduler.step(current_loss_val)

        if epoch % convergence_check_freq == 0 or current_loss_val < 0.001:
            model.eval()
            with torch.no_grad():
                rounded_preds = round_prediction(predictions)
            model.train()
            if torch.equal(rounded_preds, round_prediction(Y_target_matrix)):
                return epoch

        if current_loss_val < best_loss_val - 1e-5:
            best_loss_val = current_loss_val
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience and epoch > 50:
                model.eval()
                with torch.no_grad():
                    rounded_preds_final = round_prediction(model(X_tensor))
                if torch.equal(rounded_preds_final, round_prediction(Y_target_matrix)):
                    return epoch
                return float('inf')

        if current_loss_val < loss_convergence_threshold:
            model.eval()
            with torch.no_grad():
                rounded_preds_loss_thresh = round_prediction(model(X_tensor))
            if torch.equal(rounded_preds_loss_thresh, round_prediction(Y_target_matrix)):
                return epoch

    model.eval()
    with torch.no_grad():
        rounded_preds_max_epoch = round_prediction(model(X_tensor))
    if torch.equal(rounded_preds_max_epoch, round_prediction(Y_target_matrix)):
        return max_epochs

    return float('inf')

def evaluate_candidate_options(current_options_dict, truth_table_np, final_formula_str, bitlist_target):
    candidate_iterations_map = {}
    for idx, tree_cand_tuple_repr in current_options_dict.items():
        cand_formula_str = tree_to_formula(tree_cand_tuple_repr)
        if cand_formula_str == final_formula_str: return cand_formula_str # Exact match, return immediately
        if 'Z' not in cand_formula_str: # No Z variables, check if it's a solution
            all_rows_match = True
            for i, truth_table_row_np in enumerate(truth_table_np):
                if not check_tree_matches(tree_cand_tuple_repr, bitlist_target[i], tuple(truth_table_row_np)):
                    all_rows_match = False; break
            if all_rows_match: return cand_formula_str # It's a non-Z solution

    # If no direct match or non-Z solution found yet, evaluate Z-containing candidates
    for idx, tree_cand_tuple_repr in current_options_dict.items():
        if 'Z' not in tree_to_formula(tree_cand_tuple_repr): continue # Already checked non-Z or it's the target

        # Check if this candidate is even solvable for the first row before training
        first_row_assignments_tuple = tuple(truth_table_np[0])
        initial_candidate_list, num_Z_vars = find_allowable_combinations(
            tree_cand_tuple_repr, bitlist_target[0], first_row_assignments_tuple)

        if not initial_candidate_list and num_Z_vars > 0: # Not solvable for first row
            continue

        iterations_to_converge = train_on_truth_table(truth_table_np, bitlist_target, tree_cand_tuple_repr)
        if iterations_to_converge is not None and iterations_to_converge != float('inf'):
            candidate_iterations_map[idx] = iterations_to_converge

    return candidate_iterations_map


# Function to be executed by each worker process
def process_single_db_row(args_tuple):
    (row_db_idx, db_row_data_tuple, column_names_list,
     truth_table_np_shared, formula_idx_main, category_idx_main,
     derivation_timeout_seconds, torch_num_threads) = args_tuple

    if torch_num_threads:
        torch.set_num_threads(torch_num_threads)

    db_row_data = db_row_data_tuple
    target_minimal_formula_str = db_row_data[formula_idx_main]
    bitlist_target = binary_to_bitlist(db_row_data[category_idx_main], len(truth_table_np_shared))

    first_row_correct_output = bitlist_target[0]
    first_row_assignments_tuple = tuple(truth_table_np_shared[0])

    current_tree_candidate_tuple = None
    solution_found_for_row = False
    found_formula_str_result = None
    length_percent_diff_result = None
    depth_found_at = None
    max_depth = int(os.environ.get("MAX_SEARCH_DEPTH", "250"))


    for depth in range(max_depth):
        options_from_derivation = run_derivation_for_row(row_db_idx, db_row_data, column_names_list, current_tree_candidate_tuple)

        if not options_from_derivation:
            break

        derivation_eval_start_time = time.time()
        eval_result = evaluate_candidate_options(options_from_derivation, truth_table_np_shared,
                                                 target_minimal_formula_str, bitlist_target)
        derivation_eval_duration = time.time() - derivation_eval_start_time

        if derivation_eval_duration > derivation_timeout_seconds and depth > 0 :
            break

        if isinstance(eval_result, str):
            solution_found_for_row = True
            found_formula_str_result = eval_result
            depth_found_at = depth
            if found_formula_str_result == target_minimal_formula_str:
                length_percent_diff_result = 0.0
            else:
                length_percent_diff_result = percent_longer(found_formula_str_result, target_minimal_formula_str)
            break
        elif eval_result: # Dictionary of {option_idx: nn_iterations}
            sorted_candidates_by_iters = sorted(eval_result.items(), key=lambda item: item[1])
            chosen_next_candidate = False
            for option_idx, nn_iters in sorted_candidates_by_iters:
                potential_next_tree_tuple = options_from_derivation[option_idx]
                allowable_combs, _ = find_allowable_combinations(
                    potential_next_tree_tuple, first_row_correct_output,
                    first_row_assignments_tuple, initial_x_counter=0)
                if allowable_combs: # If it's possible to satisfy the first row
                    current_tree_candidate_tuple = potential_next_tree_tuple
                    chosen_next_candidate = True
                    break
            if not chosen_next_candidate:
                break
        else:
            break

    if solution_found_for_row:
        return "found", row_db_idx, found_formula_str_result, length_percent_diff_result, depth_found_at
    else:
        return "unfound", row_db_idx, target_minimal_formula_str, None, None


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)


    # --- SLURM Array Task Configuration ---
    py_row_start_index_str = os.environ.get("PY_ROW_START_INDEX")
    py_row_count_str = os.environ.get("PY_ROW_COUNT")

    # Get SLURM identifiers for unique file naming
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "localjob") 
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "task0")

    py_row_start_index = int(py_row_start_index_str)
    py_row_count = int(py_row_count_str)
    print(f"SLURM Task {slurm_task_id}: Received PY_ROW_START_INDEX={py_row_start_index}, PY_ROW_COUNT={py_row_count}", flush=True)


    # --- Database Connection and Data Loading ---
    DB_NAME = "sample.db"
    DB_PATH_ENV = os.environ.get("DB_PATH_OVERRIDE")
    DB_PATH = DB_PATH_ENV if DB_PATH_ENV else os.path.join(SCRIPT_DIR, DB_NAME)


    all_rows_data_main = []
    all_column_names_main = []
    formula_idx_main = -1
    category_idx_main = -1
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM data ORDER BY rowid") # Ensure consistent order
    all_rows_data_main = cursor.fetchall()
    total_db_rows_actual = len(all_rows_data_main)

    print(f"Fetched {total_db_rows_actual} total rows.", flush=True)
    all_column_names_main = [desc[0] for desc in cursor.description]
    formula_idx_main = all_column_names_main.index("formula")
    category_idx_main = all_column_names_main.index("category")


    script_start_time_main = time.time()
    truth_table_np_main = np.array([
        [1,1,1,1],[1,1,1,0],[1,1,0,1],[1,1,0,0],[1,0,1,1],[1,0,1,0],[1,0,0,1],[1,0,0,0],
        [0,1,1,1],[0,1,1,0],[0,1,0,1],[0,1,0,0],[0,0,1,1],[0,0,1,0],[0,0,0,1],[0,0,0,0]
    ], dtype=np.int8)

    # --- Determine rows for THIS SPECIFIC SLURM TASK ---
    py_row_end_index = min(py_row_start_index + py_row_count, total_db_rows_actual)
    rows_for_this_task_data = [] # Will store (original_db_idx, data_tuple)

    if py_row_start_index < total_db_rows_actual and py_row_count > 0:
        for i in range(py_row_start_index, py_row_end_index):
            rows_for_this_task_data.append({'original_db_idx': i, 'data': all_rows_data_main[i]})

    actual_rows_to_process_count_this_task = len(rows_for_this_task_data)
    if actual_rows_to_process_count_this_task > 0:
        print(f"SLURM Task {slurm_task_id}: Processing {actual_rows_to_process_count_this_task} DB rows. Original indices: {rows_for_this_task_data[0]['original_db_idx']} to {rows_for_this_task_data[-1]['original_db_idx']}", flush=True)
    else:
        print(f"SLURM Task {slurm_task_id}: No rows to process for this task.", flush=True)


    num_workers = 24
    print(f"SLURM Task {slurm_task_id}: Using {num_workers} workers for its {actual_rows_to_process_count_this_task} rows.", flush=True)
    torch_threads_per_worker = 1
    derivation_timeout_main = 1.5

    tasks_args_list = []
    for row_info in rows_for_this_task_data:
        tasks_args_list.append(
            (row_info['original_db_idx'], row_info['data'], all_column_names_main,
             truth_table_np_main, formula_idx_main, category_idx_main,
             derivation_timeout_main, torch_threads_per_worker)
        )

    csv_output_rows = []

    unfound_rows_aggregated = []
    percent_length_differences_aggregated = []
    found_solutions_count_aggregated = 0
    minimal_solutions_count_aggregated = 0
    processed_count = 0


    print(f"SLURM Task {slurm_task_id}: Starting ProcessPoolExecutor with {num_workers} workers for {len(tasks_args_list)} DB rows.", flush=True)
    if tasks_args_list:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_task_args = {executor.submit(process_single_db_row, args): args for args in tasks_args_list}

            for future in concurrent.futures.as_completed(future_to_task_args):
                task_args = future_to_task_args[future]
                original_db_idx = task_args[0]
                original_row_tuple_data = task_args[1] 

                # Prepare data for CSV
                row_for_csv = {col_name: original_row_tuple_data[i] for i, col_name in enumerate(all_column_names_main)}
                row_for_csv['original_db_idx_processed'] = original_db_idx
                row_for_csv['script_found_formula'] = None
                row_for_csv['script_overlength_percent'] = None
                row_for_csv['depth_found'] = None

                # MODIFIED: Unpack depth from result
                status, _, data1, data2, found_at_depth = future.result()

                if status == "found":
                    found_formula_str = data1
                    length_pct_diff = data2
                    row_for_csv['script_found_formula'] = found_formula_str
                    row_for_csv['script_overlength_percent'] = length_pct_diff
                    row_for_csv['depth_found'] = found_at_depth

                    found_solutions_count_aggregated += 1
                    percent_length_differences_aggregated.append(length_pct_diff)
                    if length_pct_diff == 0.0:
                        minimal_solutions_count_aggregated += 1
                elif status == "unfound":
                    target_minimal_formula = data1
                    unfound_rows_aggregated.append((original_db_idx, target_minimal_formula))


                csv_output_rows.append(row_for_csv)
                processed_count += 1
                if processed_count % (max(1, len(tasks_args_list) // 10)) == 0 or processed_count == len(tasks_args_list):
                    print(f"  SLURM Task {slurm_task_id} Progress: {processed_count}/{len(tasks_args_list)} rows processed.", flush=True)
    else:
        print(f"SLURM Task {slurm_task_id}: No tasks to process.", flush=True)

    csv_output_rows.sort(key=lambda x: x['original_db_idx_processed'])


    if csv_output_rows:
        output_dir = "task_results" # Subdirectory for individual task CSVs
        os.makedirs(output_dir, exist_ok=True)

        # Filename: results_arrayJOBID_taskTASKID.csv
        csv_filename = os.path.join(output_dir, f"results_{slurm_array_job_id}_task_{slurm_task_id}.csv")

        fieldnames = all_column_names_main + ['script_found_formula', 'script_overlength_percent', 'depth_found']

        print(f"SLURM Task {slurm_task_id}: Writing {len(csv_output_rows)} rows to {csv_filename}", flush=True)
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row_dict in csv_output_rows:
                writer.writerow(row_dict)
        print(f"SLURM Task {slurm_task_id}: Successfully wrote results to {csv_filename}", flush=True)


    # --- Console Summary (for this task's log) ---
    total_execution_time_main = time.time() - script_start_time_main
    print(f"\nSLURM Task {slurm_task_id}: Total script execution time: {total_execution_time_main:.2f} seconds.", flush=True)
    print(f"\n\n=== Summary for SLURM Task {slurm_task_id} (from {actual_rows_to_process_count_this_task} assigned rows) ===", flush=True)
    print(f"  Actual rows processed by this task's workers: {processed_count}", flush=True)
    print(f"  Rows with *any* formula found by this task: {found_solutions_count_aggregated}", flush=True)
    print(f"  Of those, exact minimals found by this task: {minimal_solutions_count_aggregated}", flush=True)

    if percent_length_differences_aggregated:
        avg_pct_len_diff = sum(percent_length_differences_aggregated) / len(percent_length_differences_aggregated)
        print(f"  Avg %-overlength (this task): {avg_pct_len_diff:.1f}%", flush=True)

    print(f"\n--- Found Formula Details (SLURM Task {slurm_task_id}) ---", flush=True)
    found_formulas_details_print = []
    for row in csv_output_rows:
        if row.get("script_found_formula") and row.get("script_found_formula") not in [None, "ERROR_IN_PROCESSING"]:
            found_formulas_details_print.append(
                (row['original_db_idx_processed'], row["script_found_formula"], row["script_overlength_percent"], row.get('depth_found'))
            )
    found_formulas_details_print.sort()
    for db_idx, formula_str, pct_diff, depth_val in found_formulas_details_print:
        depth_str = f"depth {depth_val}" if depth_val is not None else "depth N/A"
        print(f"  DB Row {db_idx}: {formula_str!r} ({pct_diff:+.1f}%, {depth_str})", flush=True)

    if unfound_rows_aggregated:
        print(f"\n--- Rows Not Found or Errored (SLURM Task {slurm_task_id}, {len(unfound_rows_aggregated)}) ---", flush=True)
        unfound_rows_aggregated.sort()
        for db_idx, status_or_formula in unfound_rows_aggregated:
            if status_or_formula == "ERROR_IN_PROCESSING":
                 print(f"  DB Row {db_idx}: ERRORED DURING PROCESSING", flush=True)
            else: 
                 print(f"  DB Row {db_idx}: Minimal formula was {status_or_formula!r}", flush=True)

    print(f"SLURM Task {slurm_task_id} finished.", flush=True)