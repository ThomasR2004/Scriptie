import torch
from functools import lru_cache

class TreeNode:
    def __init__(self, op, left=None, right=None):
        self.op = op
        self.left = left
        self.right = right
        self._hash = None

    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return NotImplemented
        return self.op == other.op and self.left == other.left and self.right == other.right

    def __hash__(self):
        if self._hash is None:
            self._hash = hash((self.op, self.left, self.right))
        return self._hash

    def __repr__(self):
        if self.left is None and self.right is None:
            return f"TreeNode({self.op!r})"
        return f"TreeNode({self.op!r}, left={self.left!r}, right={self.right!r})"

TREENODE_CACHE = {} 
def tuple_to_treenode(tree_repr):
    if isinstance(tree_repr, TreeNode):
        return tree_repr
    if isinstance(tree_repr, str):
        return TreeNode(tree_repr)
    if isinstance(tree_repr, list): tree_repr = tuple(tree_repr)
    elif isinstance(tree_repr, tuple):
        tree_repr = tuple(tuple_to_treenode(arg) if isinstance(arg, (list, tuple, str)) and arg not in ['N', 'A', 'O', 'C', 'NC', 'B', 'X', 'NA', 'NOR'] else arg for arg in tree_repr)


    if tree_repr in TREENODE_CACHE:
        return TREENODE_CACHE[tree_repr]
    
    op = tree_repr[0] 
    args = tree_repr[1:]

    if op == 'N':
        if not args: raise ValueError(f"Negation 'N' operator expects 1 argument, got 0 in {tree_repr}")
        left_child = tuple_to_treenode(args[0])
        result = TreeNode('N', left=left_child)
    else: # Binary operators
        if len(args) < 2: raise ValueError(f"Binary operator '{op}' expects 2 arguments, got {len(args)} in {tree_repr}")
        left_child = tuple_to_treenode(args[0])
        right_child = tuple_to_treenode(args[1])
        result = TreeNode(op, left=left_child, right=right_child)
    
    TREENODE_CACHE[tree_repr] = result
    return result


TRUTH_TABLES = {
    'A': {1: [(1, 1)], 0: [(0, 0), (0, 1), (1, 0)]}, 'O': {1: [(1, 1), (1, 0), (0, 1)], 0: [(0, 0)]},
    'C': {1: [(0, 0), (0, 1), (1, 1)], 0: [(1, 0)]}, 'NC': {1: [(1, 0)], 0: [(0, 0), (0, 1), (1, 1)]},
    'B': {1: [(1, 1), (0, 0)], 0: [(1, 0), (0, 1)]}, 'X': {1: [(1, 0), (0, 1)], 0: [(1, 1), (0, 0)]},
    'NA': {1: [(0, 0), (0, 1), (1, 0)], 0: [(1, 1)]}, 'NOR': {1: [(0, 0)], 0: [(1, 1), (1, 0), (0, 1)]}
}

def _find_combinations(node, correct_output, assignments, current_x_counter):
    p_val, q_val, r_val, s_val = assignments
    var_map = {'p': p_val, 'q': q_val, 'r': r_val, 's': s_val}
    op = node.op
    if op in var_map:
        return ([{}] if var_map[op] == correct_output else []), current_x_counter
    if op == 'Z':
        return ([{current_x_counter: correct_output}]), current_x_counter + 1
    if op == 'N':
        return _find_combinations(node.left, 1 - correct_output, assignments, current_x_counter)
    if node.left is None or node.right is None:
        raise ValueError(f"Binary operator {op} node is missing children: {node}")
    
    if op not in TRUTH_TABLES: # Defensive check
        raise ValueError(f"Operator '{op}' not found in TRUTH_TABLES for node {node}")
    if correct_output not in TRUTH_TABLES[op]: # Defensive check
        raise ValueError(f"Correct output '{correct_output}' not valid for operator '{op}' in TRUTH_TABLES for node {node}")
        
    possible_input_pairs = TRUTH_TABLES[op][correct_output]
    all_combined_results = []
    max_x_after_this_node = current_x_counter
    for left_val, right_val in possible_input_pairs:
        left_branch_options, x_after_left = _find_combinations(node.left, left_val, assignments, current_x_counter)
        max_x_after_this_node = max(max_x_after_this_node, x_after_left)
        if not left_branch_options: continue
        right_branch_options, x_after_right = _find_combinations(node.right, right_val, assignments, x_after_left)
        max_x_after_this_node = max(max_x_after_this_node, x_after_right)
        if not right_branch_options: continue
        combined_for_this_pair = combine(left_branch_options, right_branch_options)
        all_combined_results.extend(combined_for_this_pair)
    return all_combined_results, max_x_after_this_node

@lru_cache(maxsize=1000)
def find_allowable_combinations(tree_tuple_repr, correct_output, assignments_tuple, initial_x_counter=0):
    node = tuple_to_treenode(tree_tuple_repr)
    return _find_combinations(node, correct_output, assignments_tuple, initial_x_counter)


def combine(l1, l2):
    if not l1 or not l2:
        return []
    
    # Ensure smaller list is inner loop
    if len(l1) < len(l2):
        shorter, longer = l1, l2
        swap = False
    else:
        shorter, longer = l2, l1
        swap = True
    
    # Pre-filter and convert to items for faster comparison
    shorter_valid = []
    for d in shorter:
        if isinstance(d, dict):
            shorter_valid.append((d, frozenset(d.items())))
    
    longer_valid = []
    for d in longer:
        if isinstance(d, dict):
            longer_valid.append((d, frozenset(d.items())))
    
    if not shorter_valid or not longer_valid:
        return []
    
    res = []
    
    for d_small, items_small in shorter_valid:
        for d_large, items_large in longer_valid:
            # Check compatibility using set intersection
            if not (items_small & items_large - items_small):
                # Compatible - merge dictionaries
                if swap:
                    merged = {**d_large, **d_small}
                else:
                    merged = {**d_small, **d_large}
                res.append(merged)
    
    return res
 


def round_prediction(tensor):
    return (tensor >= 0.5).to(torch.float)


def binary_to_bitlist(n, total):
    return [int(a) for a in f'{n:0{total}b}']
    
def tree_to_formula(tree):
    if not isinstance(tree, tuple):
        return str(tree)

    op, *children = tree
    rendered_children = [tree_to_formula(c) for c in children]
    return f"{op}({','.join(rendered_children)})"


def extract_grammar_from_data_row(row, columns):
    operator_names = {
        "A": "A",
        "O": "O",
        "C": "C",
        "NC": "NC",
        "B": "B",
        "X": "X",
        "NA": "NA",
        "NOR": "NOR",
        "N": "N",
    }

    grammar = {}
    for op, name in operator_names.items():
        if op in columns and row[columns.index(op)] == 1:
            if op == "N":
                grammar[op] = lambda Z, name=name: (name, Z)
            else:
                grammar[op] = lambda Z, name=name: (name, Z, Z)
    return grammar



def count_non_terminals(expr):
    if expr == "Z":
        return 1
    elif isinstance(expr, tuple):
        return sum(count_non_terminals(sub) for sub in expr)
    return 0

def expand_all_X(expr, grammar, max_non_terminals=8):
    if expr == "Z":
        expansions = []

        # Always allow terminal replacements
        for terminal in ["p", "q", "r", "s"]:
            expansions.append(terminal)

        for rule in grammar.values():
            new_expr = rule("Z")
            total_Zs = count_non_terminals(new_expr)
            if total_Zs <= max_non_terminals:
                expansions.append(new_expr)

        return expansions

    elif isinstance(expr, tuple):
        for i, sub in enumerate(expr):
            sub_expansions = expand_all_X(sub, grammar, max_non_terminals)
            if sub_expansions:
                results = []
                for new_sub in sub_expansions:
                    new_expr = list(expr)
                    new_expr[i] = new_sub
                    combined = tuple(new_expr)
                    if count_non_terminals(combined) <= max_non_terminals:
                        results.append(combined)
                return results

    return []



def run_derivation_for_row(row_idx, row, columns, current = None):
    grammar = extract_grammar_from_data_row(row, columns)
    
    if current is None:
        current = "Z"

    # Get all possible expansions for the current expression
    expansions = expand_all_X(current, grammar)
    if not expansions:
        return current, {}

    # Build an options dictionary mapping indices to expansion expressions
    current_options = {i: exp for i, exp in enumerate(expansions)}
    
    #print(f"Options dict:\n  {current_options}")

    return current_options

def evaluate_tree(tree, assignments):
    # Debugging: Print tree and assignments at each call
    #print(f"Evaluating tree: {tree} with assignments: {assignments}")

    if isinstance(tree, (int, bool)):
        return int(tree)  # Directly return 1 or 0
    if isinstance(tree, str):
        if tree == 'a':
            return 1
        if tree == 'b':
            return 0
        if tree == 'p':
            return assignments[0]
        if tree == 'q':
            return assignments[1]
        if tree == 'r':
            return assignments[2]
        if tree == 's':
            return assignments[3]
        raise ValueError(f"Unknown leaf: {tree!r}")

    # 2) The tree must be a tuple for operators
    if not isinstance(tree, tuple):
        raise ValueError(f"Invalid tree node: {tree!r}")

    op, *args = tree

    # 3) Operators: Check each operation type
    if op == 'N':
        assert len(args) == 1
        return 1 - evaluate_tree(args[0], assignments)

    if op == 'A':
        assert len(args) == 2
        return evaluate_tree(args[0], assignments) & evaluate_tree(args[1], assignments)

    if op == 'O':
        assert len(args) == 2
        return evaluate_tree(args[0], assignments) | evaluate_tree(args[1], assignments)

    if op == 'C':  
        assert len(args) == 2
        left = evaluate_tree(args[0], assignments)
        right = evaluate_tree(args[1], assignments)
        return 0 if (left == 1 and right == 0) else 1

    if op == 'NC': 
        assert len(args) == 2
        left = evaluate_tree(args[0], assignments)
        right = evaluate_tree(args[1], assignments)
        return 1 if (left == 1 and right == 0) else 0

    if op == 'B':  
        assert len(args) == 2
        return int(evaluate_tree(args[0], assignments) == evaluate_tree(args[1], assignments))

    if op == 'X':  
        assert len(args) == 2
        return int(evaluate_tree(args[0], assignments) != evaluate_tree(args[1], assignments))

    if op == 'NA': 
        assert len(args) == 2
        return 1 - (evaluate_tree(args[0], assignments) & evaluate_tree(args[1], assignments))

    if op == 'NOR':  
        assert len(args) == 2
        return 1 - (evaluate_tree(args[0], assignments) | evaluate_tree(args[1], assignments))

    raise ValueError(f"Unknown operator or malformed node: {op!r}")


    
    
def check_tree_matches(tree, target, assignments):
    val = evaluate_tree(tree, assignments)
    return bool(val == target)
def percent_longer(found: str, minimal: str) -> float:
    return (len(found) - len(minimal)) / len(minimal) * 100






