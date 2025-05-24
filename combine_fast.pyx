# combine_fast.pyx
# Cython implementation for maximum performance

import cython
from libc.stdlib cimport malloc, free

@cython.boundscheck(False)
@cython.wraparound(False)
def combine_cython(list l1, list l2):
    """Cython-optimized combine function with C-level performance."""
    cdef int len1 = len(l1)
    cdef int len2 = len(l2)
    
    if len1 == 0 or len2 == 0:
        return []
    
    cdef:
        list shorter, longer
        list res = []
        int i, j
        dict d_small, d_large, merged
        bint swap, compatible
        object key, val
    
    # Ensure smaller list is inner loop
    if len1 < len2:
        shorter = l1
        longer = l2
        swap = False
    else:
        shorter = l2
        longer = l1
        swap = True
    
    # Main loop with C-level indexing
    for i in range(len(shorter)):
        d_small = shorter[i]
        if type(d_small) is not dict:
            continue
            
        for j in range(len(longer)):
            d_large = longer[j]
            if type(d_large) is not dict:
                continue
            
            # Fast compatibility check
            compatible = True
            for key in d_small:
                if key in d_large:
                    if d_large[key] != d_small[key]:
                        compatible = False
                        break
            
            if compatible:
                if swap:
                    merged = dict(d_large)
                    merged.update(d_small)
                else:
                    merged = dict(d_small)
                    merged.update(d_large)
                res.append(merged)
    
    return res


@cython.boundscheck(False)
@cython.wraparound(False)
def combine_cython_ultra(list l1, list l2):
    """Ultra-optimized Cython version with pre-filtering."""
    cdef int len1 = len(l1)
    cdef int len2 = len(l2)
    
    if len1 == 0 or len2 == 0:
        return []
    
    cdef:
        list shorter_valid = []
        list longer_valid = []
        list shorter, longer
        list res = []
        int i, j, num_shorter, num_longer
        dict d_small, d_large, merged
        bint swap, compatible
        object key
    
    # Pre-filter valid dictionaries
    if len1 < len2:
        shorter = l1
        longer = l2
        swap = False
    else:
        shorter = l2
        longer = l1
        swap = True
    
    # Filter to valid dicts only
    for i in range(len(shorter)):
        if type(shorter[i]) is dict:
            shorter_valid.append(shorter[i])
    
    for i in range(len(longer)):
        if type(longer[i]) is dict:
            longer_valid.append(longer[i])
    
    num_shorter = len(shorter_valid)
    num_longer = len(longer_valid)
    
    if num_shorter == 0 or num_longer == 0:
        return []
    
    # Main processing loop
    for i in range(num_shorter):
        d_small = shorter_valid[i]
        
        for j in range(num_longer):
            d_large = longer_valid[j]
            
            # Compatibility check with early exit
            compatible = True
            for key in d_small:
                if key in d_large and d_large[key] != d_small[key]:
                    compatible = False
                    break
            
            if compatible:
                if swap:
                    merged = d_large.copy()
                    merged.update(d_small)
                else:
                    merged = d_small.copy()
                    merged.update(d_large)
                res.append(merged)
    
    return res
