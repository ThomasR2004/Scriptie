

def find_allowable_combinations(tree, correct, assignments, x_counter=0):
    p, q, r, s = assignments

    f, *args = tree
    op = f


    # A helper to union results (for alternatives) while threading the counter.
    def union_results(results):
        union_list = []
        max_counter = x_counter  # starting counter
        for res, cnt in results:
            union_list.extend(res)
            max_counter = max(max_counter, cnt)
        return union_list, max_counter

    # A helper to combine two lists of constraint dictionaries.
    def combine(list1, list2):
        result = []
        for d1 in list1:
            for d2 in list2:
                merged = dict(d1)  # Make sure d1 is treated as a dict
                conflict = False
                for key, value in d2.items():
                    if key in merged and merged[key] != value:
                        conflict = True
                        break
                    merged[key] = value
                if not conflict:
                    result.append(merged)
        return result


    # Process operators.
    if op == 'N':
        res, new_counter = find_allowable_combinations(args[0], 1 - correct, assignments, x_counter)
        return res, new_counter

    elif op == 'A':
        if correct == 1:
            left, counter_left = find_allowable_combinations(args[0], 1, assignments, x_counter)
            right, counter_right = find_allowable_combinations(args[1], 1, assignments, counter_left)
            return combine(left, right), counter_right
        else:
            branch1_left, counter1 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 1, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 0, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            branch3_left, counter3 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch3_right, counter3 = find_allowable_combinations(args[1], 0, assignments, counter3)
            poss3 = combine(branch3_left, branch3_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2), (poss3, counter3)
            ])
            return union_list, final_counter

    elif op == 'O':
        if correct == 1:
            branch1_left, counter1 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 1, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 0, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            branch3_left, counter3 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch3_right, counter3 = find_allowable_combinations(args[1], 1, assignments, counter3)
            poss3 = combine(branch3_left, branch3_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2), (poss3, counter3)
            ])
            return union_list, final_counter
        else:
            left, counter_left = find_allowable_combinations(args[0], 0, assignments, x_counter)
            right, counter_right = find_allowable_combinations(args[1], 0, assignments, counter_left)
            return combine(left, right), counter_right

    elif op == 'C':
        if correct == 1:
            branch1_left, counter1 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 0, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 1, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            branch3_left, counter3 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch3_right, counter3 = find_allowable_combinations(args[1], 1, assignments, counter3)
            poss3 = combine(branch3_left, branch3_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2), (poss3, counter3)
            ])
            return union_list, final_counter
        else:
            left, counter_left = find_allowable_combinations(args[0], 1, assignments, x_counter)
            right, counter_right = find_allowable_combinations(args[1], 0, assignments, counter_left)
            return combine(left, right), counter_right

    elif op == 'NC':
        if correct == 1:
            left, counter_left = find_allowable_combinations(args[0], 1, assignments, x_counter)
            right, counter_right = find_allowable_combinations(args[1], 0, assignments, counter_left)
            return combine(left, right), counter_right
        else:
            branch1_left, counter1 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 0, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 1, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            branch3_left, counter3 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch3_right, counter3 = find_allowable_combinations(args[1], 1, assignments, counter3)
            poss3 = combine(branch3_left, branch3_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2), (poss3, counter3)
            ])
            return union_list, final_counter

    elif op == 'B':
        if correct == 1:
            branch1_left, counter1 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 1, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 0, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2)
            ])
            return union_list, final_counter
        else:
            branch1_left, counter1 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 0, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 1, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2)
            ])
            return union_list, final_counter

    elif op == 'X':
        if correct == 1:
            branch1_left, counter1 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 0, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 1, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2)
            ])
            return union_list, final_counter
        else:
            branch1_left, counter1 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 1, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 0, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2)
            ])
            return union_list, final_counter

    elif op == 'NA':
        if correct == 1:
            branch1_left, counter1 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 0, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 1, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            branch3_left, counter3 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch3_right, counter3 = find_allowable_combinations(args[1], 0, assignments, counter3)
            poss3 = combine(branch3_left, branch3_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2), (poss3, counter3)
            ])
            return union_list, final_counter
        else:
            left, counter_left = find_allowable_combinations(args[0], 1, assignments, x_counter)
            right, counter_right = find_allowable_combinations(args[1], 1, assignments, counter_left)
            return combine(left, right), counter_right

    elif op == 'NOR':
        if correct == 1:
            left, counter_left = find_allowable_combinations(args[0], 0, assignments, x_counter)
            right, counter_right = find_allowable_combinations(args[1], 0, assignments, counter_left)
            return combine(left, right), counter_right
        else:
            branch1_left, counter1 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch1_right, counter1 = find_allowable_combinations(args[1], 0, assignments, counter1)
            poss1 = combine(branch1_left, branch1_right)

            branch2_left, counter2 = find_allowable_combinations(args[0], 0, assignments, x_counter)
            branch2_right, counter2 = find_allowable_combinations(args[1], 1, assignments, counter2)
            poss2 = combine(branch2_left, branch2_right)

            branch3_left, counter3 = find_allowable_combinations(args[0], 1, assignments, x_counter)
            branch3_right, counter3 = find_allowable_combinations(args[1], 1, assignments, counter3)
            poss3 = combine(branch3_left, branch3_right)

            union_list, final_counter = union_results([
                (poss1, counter1), (poss2, counter2), (poss3, counter3)
            ])
            return union_list, final_counter

    elif op in ('p', 'q', 'r', 's'):
        # Check if the predetermined value agrees with the expected
        val = {'p': p, 'q': q, 'r': r, 's': s}[op]
        return ([] if val != correct else [{}]), x_counter

    elif op == 'Z':
        # For an unknown "X" node, assign a unique variable name.
        new_var = f"Z_{x_counter}"
        return ([{new_var: correct}]), x_counter + 1
    return
 


def round_prediction(pred, threshold=0.5):
    return (pred > threshold).float()


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



def expand_all_X(expr, grammar):
    """
    Recursively finds the leftmost 'X' in a nested tuple structure and replaces it
    with each possible grammar rule or terminal symbol.
    """
    if expr == "Z":
        # Base case: single 'X' to replace
        expansions = []
        
        for terminal in ["p", "q", "r", "s"]:
            expansions.append(terminal)

        for rule in grammar.values():
            expansions.append(rule("Z"))

        return expansions

    elif isinstance(expr, tuple):
        # Recursive case: traverse the structure to find the leftmost 'X'
        for i, sub in enumerate(expr):
            sub_expansions = expand_all_X(sub, grammar)
            if sub_expansions:
                # Replace the first expandable part and break
                results = []
                for new_sub in sub_expansions:
                    new_expr = list(expr)
                    new_expr[i] = new_sub
                    results.append(tuple(new_expr))
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


