import torch

def find_allowable_combinations(tree, correct, assignments, x_counter=0):
    p, q, r, s = assignments
    
    value_map = {'p': p, 'q': q, 'r': r, 's': s}
    
    # Pre-computed truth tables for common operations
    truth_tables = {
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


    # Extract operator and arguments
    f, *args = tree
    op = f
    
    # First handle terminal cases (most common)
    if op in ('p', 'q', 'r', 's'):
        val = value_map[op]
        return ([] if val != correct else [{}]), x_counter
        
    elif op == 'Z':
        new_var = f"Z_{x_counter}"
        return ([{new_var: correct}]), x_counter + 1
        
    # Handle NOT separately (common case)
    elif op == 'N':
        return find_allowable_combinations(args[0], 1 - correct, assignments, x_counter)
    
    # Process binary operators more efficiently using truth tables
    elif op in truth_tables:
        valid_combinations = truth_tables[op][correct]
        results = []
        final_counter = x_counter
        
        for left_val, right_val in valid_combinations:
            left_results, temp_counter = find_allowable_combinations(args[0], left_val, assignments, x_counter)
            
            # Skip if left side has no solutions
            if not left_results:
                final_counter = max(final_counter, temp_counter)
                continue
                
            right_results, right_counter = find_allowable_combinations(args[1], right_val, assignments, temp_counter)
            
            # Skip if right side has no solutions
            if not right_results:
                final_counter = max(final_counter, right_counter)
                continue
                
            # Combine valid solutions
            combined = combine(left_results, right_results)
            results.extend(combined)
            final_counter = max(final_counter, right_counter)
            
        return results, final_counter
    
    # Fallback case (should not reach here if all operators are handled)
    return [], x_counter
 


def round_prediction(tensor):
    return (tensor >= 0.5).to(torch.float)


def binary_to_bitlist(n, total):
    return [int(a) for a in f'{n:0{total}b}']
    
def tree_to_formula(tree):
    """
    Recursively render a nested tuple/tree
    into a string like "NA(q,s)" or "AND(p,OR(q,r))".
    """
    if not isinstance(tree, tuple):
        return str(tree)

    op, *children = tree
    rendered_children = [tree_to_formula(c) for c in children]
    return f"{op}({','.join(rendered_children)})"


def extract_grammar_from_data_row(row, columns):
    """
    Build grammar from a row in the 'data' table.
    Each rule returns a nested tuple.
    """
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
    """Recursively count the number of non-terminal symbols ('Z') in the expression."""
    if expr == "Z":
        return 1
    elif isinstance(expr, tuple):
        return sum(count_non_terminals(sub) for sub in expr)
    return 0

def expand_all_X(expr, grammar, max_non_terminals=8):
    """
    Recursively finds the leftmost 'Z' in a nested tuple structure and replaces it
    with each possible grammar rule or terminal symbol. Grammar rules are only
    applied if the resulting expression doesn't exceed the non-terminal threshold.
    """
    if expr == "Z":
        expansions = []

        # Always allow terminal replacements
        for terminal in ["p", "q", "r", "s"]:
            expansions.append(terminal)

        # Try rule-based replacements, but only keep them if they don't exceed the limit
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
    """
    Expands from the starting expression 'X' using grammar derived from the row.
    This version performs a single iteration and returns the current expression
    along with the options dictionary. An external function can use the options dict
    to select the next node.

    Returns:
        current (str): The starting expression (or new node if already set).
        current_options (dict): Dictionary of expansion options indexed by integers.
    """
    print(f"Using row {row_idx}: {row}")
    grammar = extract_grammar_from_data_row(row, columns)
    
    if current is None:
        current = "Z"

    # Get all possible expansions for the current expression
    expansions = expand_all_X(current, grammar)
    if not expansions:
        print("No expansions available.")
        return current, {}

    # Build an options dictionary mapping indices to expansion expressions
    current_options = {i: exp for i, exp in enumerate(expansions)}
    
    print(f"\nCurrent expression: {current}")
    #print(f"Options dict:\n  {current_options}")

    return current_options

def evaluate_tree(tree, assignments):
    """
    Recursively evaluates a Boolean‐formula tree under the given assignments.
    Supports:
      - tuple nodes with ops: 'N','A','O','C','NC','B','X','NA','NOR'
      - variable leaves: 'p','q','r','s'
      - constant leaves: 'a' (True), 'b' (False)
    Always returns 0 or 1, or raises ValueError on malformed input.
    """
    # Debugging: Print tree and assignments at each call
    #print(f"Evaluating tree: {tree} with assignments: {assignments}")

    # 1) Constant leaf (a = True, b = False)
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

    if op == 'C':  # A → B is ¬A ∨ B
        assert len(args) == 2
        left = evaluate_tree(args[0], assignments)
        right = evaluate_tree(args[1], assignments)
        return 0 if (left == 1 and right == 0) else 1

    if op == 'NC':  # A ↛ B
        assert len(args) == 2
        left = evaluate_tree(args[0], assignments)
        right = evaluate_tree(args[1], assignments)
        return 1 if (left == 1 and right == 0) else 0

    if op == 'B':  # A ↔ B
        assert len(args) == 2
        return int(evaluate_tree(args[0], assignments) == evaluate_tree(args[1], assignments))

    if op == 'X':  # A ↮ B
        assert len(args) == 2
        return int(evaluate_tree(args[0], assignments) != evaluate_tree(args[1], assignments))

    if op == 'NA':  # ¬(A ∧ B)
        assert len(args) == 2
        return 1 - (evaluate_tree(args[0], assignments) & evaluate_tree(args[1], assignments))

    if op == 'NOR':  # ¬(A ∨ B)
        assert len(args) == 2
        return 1 - (evaluate_tree(args[0], assignments) | evaluate_tree(args[1], assignments))

    # 4) Invalid operator or malformed node
    raise ValueError(f"Unknown operator or malformed node: {op!r}")


    
    
def check_tree_matches(tree, target, assignments):
    """
    Returns True if evaluating `tree` under `assignments` yields `target`, else False.
    - tree: partial Boolean‐formula tree.
    - target: 0 or 1, the boolean you want to check against.
    - assignments: tuple (p,q,r,s).
    """
    val = evaluate_tree(tree, assignments)
    return bool(val == target)
def percent_longer(found: str, minimal: str) -> float:
    """
    Returns how much longer `found` is than `minimal`, 
    as a percentage of `minimal`’s length.
    """
    return (len(found) - len(minimal)) / len(minimal) * 100






