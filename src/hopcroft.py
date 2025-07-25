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

    As showin in Appendix A.3: Algorithm 1 Hopcroft’s Algorithm

    Input: set of states Q with output 0, set of states F with output 1
    Output: minimal state partition P
    P := {F, Q \ F}
    W := {F, Q \ F}
    while W is not empty do
        choose and remove a set A from W
            for c in Σ do
                let X be the set of states for which a transition on c leads to a state in A
                for set Y in P for which X ∩ Y is nonempty and Y \ X is nonempty do
                    replace Y in P by the two sets X ∩ Y and Y \ X
                    if Y is in W then
                        replace Y in W by the same two sets
                    else
                        if |X ∩ Y | ≤ |Y \ X| then
                            add X ∩ Y to W
                        else
                            add Y \ X to W
                        end if
                    end if
                end for
            end for
        end while
    return P

    """

    # 1) Initial partition by output value
    groups = {}
    for q, y in outputs.items():
        groups.setdefault(y, set()).add(q)
    P = [block for block in groups.values() if block]
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
