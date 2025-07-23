def hopcroft_minimize(Q, Sigma, delta, outputs):
    """Minimize a DFA/Moore machine using Hopcroft-style partition refinement.
    Initial partition is based on the output value g(q) instead of accepting/non-accepting.
    Args:
        Q: iterable of states
        Sigma: iterable alphabet symbols
        delta: dict mapping (q,a)->q'
        outputs: dict mapping q -> discrete output label (e.g. 0/1)
    Returns:
        new_states, Sigma, new_delta, new_outputs, state_map
    """
    # 1) Initial partition by output value
    groups = {}
    for q, y in outputs.items():
        groups.setdefault(y, set()).add(q)
    P = [block for block in groups.values() if block]  # list of sets
    W = [b.copy() for b in P]

    # 2) Build inverse transitions for speed
    inv = {a: {q: set() for q in Q} for a in Sigma}
    for (q, a), qn in delta.items():
        inv[a][qn].add(q)

    # 3) Refinement loop
    while W:
        A = W.pop()
        for a in Sigma:
            X = set().union(*(inv[a][q] for q in A))
            newP = []
            for Y in P:
                inter, diff = Y & X, Y - X
                if inter and diff:
                    newP.extend([inter, diff])
                    if Y in W:
                        W.remove(Y)
                        W.extend([inter, diff])
                    else:
                        # add the smaller part to W
                        W.append(inter if len(inter) <= len(diff) else diff)
                else:
                    newP.append(Y)
            P = newP

    # 4) Rebuild minimized machine
    state_map = {q: i for i, block in enumerate(P) for q in block}
    new_states = list(range(len(P)))
    new_delta = {(state_map[q], a): state_map[delta[(q, a)]]
                 for q in Q for a in Sigma}
    new_outputs = {state_map[q]: outputs[q] for q in Q}
    return new_states, Sigma, new_delta, new_outputs, state_map
