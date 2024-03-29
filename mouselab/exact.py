from toolz import memoize, unique

def sort_tree(env, state):
    """Breaks symmetry between belief states.

    This is done by enforcing that the knowledge about states at each
    depth be sorted by [0, 1, UNKNOWN]
    """
    state = list(state)
    for i in range(len(env.tree) - 1, -1, -1):
        if not env.tree[i]:
            continue
        c1, c2 = env.tree[i]
        idx1, idx2 = env.subtree_slices[c1], env.subtree_slices[c2]

        if not (state[idx1] <= state[idx2]):
            state[idx1], state[idx2] = state[idx2], state[idx1]
    return tuple(state)


def hash_tree(env, state, action=None):
    """
    Breaks symmetry between belief states.

    env: MouselabEnv or child
        last_action_info: array of size number of actions x number of actions
        include_last_action: whether last action is in state
    state: state
    action: action for hash (optional)
    """
    if state == "__term_state__":
        return hash(state)

    # include some info about last action
    if env.include_last_action and action is not env.term_action:
        # no last_action_info -> just one hot encoding of last action
        if env.last_action_info is None:
            node_last_clicks = [0 for action_idx, action in enumerate(state[:-1])]
            node_last_clicks[state[-1]] = 1
        # otherwise take the entry in that array for the last action (e.g. distances to other nodes)
        else:
            node_last_clicks = env.last_action_info[state[-1]]
        # zip up state and node last clicks
        state = [*zip(state[:-1],node_last_clicks)]

    # handles converting this to a (s,a) hash rather than just s hash
    if action is not None and action is not env.term_action:
        actions = [0 for _ in state]
        actions[action] = 1
        if env.include_last_action:
            state = [tuple([*curr_state, action]) for curr_state, action in zip(state, actions)]
        else:
            state = [*zip(state, actions)]

    def rec(n):
        x = hash(state[n])
        childs = sum(rec(c) for c in env.tree[n])
        return hash(str(x + childs))

    return rec(0)

def solve(env, hash_state=None, actions=None, blinkered=None):
    """Returns Q, V, pi, and computation data for an mdp environment."""
    info = {"q": 0, "v": 0}  # track number of times each function is called

    if hash_state is None:
        if hasattr(env, "n_arm"):
            hash_state = lambda state: tuple(sorted(state))
        elif hasattr(env, "tree"):
            # hash_state = lambda state: sort_tree(env, state)
            hash_state = lambda state: hash_tree(env, state)

    # for tests with no hashing
    if hash_state == "test":
        hash_state = None

    if actions is None:
        actions = env.actions
    if blinkered == "recursive":

        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], *env.path_to(a)[:-1], env.term_action)

    elif blinkered == "children":

        def subset_actions(a):
            if a == env.term_action:
                return ()
            return (*env.subtree[a][1:], env.term_action)

    elif blinkered == "branch":
        assert hasattr(env, "_relevant_subtree")

        def subset_actions(a):
            if a == env.term_action:
                return ()
            else:
                return (*env._relevant_subtree(a), env.term_action)

    elif blinkered:

        def subset_actions(a):
            return (a, env.term_action)

    else:
        subset_actions = lambda a: None

    if hash_state is not None:

        def hash_key(args, kwargs):
            state = args[0]
            if state is None:
                return state
            else:
                if kwargs:
                    # Blinkered approximation. Hash key is insensitive
                    # to states that can't be acted on, except for the
                    # best expected value
                    # Embed the action subset into the state.
                    action_subset = kwargs["action_subset"]
                    mask = [0] * len(state)
                    for a in action_subset:
                        mask[a] = 1
                    state = tuple(zip(state, mask))
                return hash_state(state)

    else:
        hash_key = None

    @memoize
    def Q(s, a):
        info["q"] += 1
        action_subset = subset_actions(a)
        return sum(p * (r + V(s1, action_subset)) for p, s1, r in env.results(s, a))

    @memoize(key=hash_key)
    def V(s, action_subset=None):
        if s is None:
            return 0
        info["v"] += 1
        acts = actions(s)
        if action_subset is not None:
            acts = tuple(a for a in acts if a in action_subset)
        return max((Q(s, a) for a in acts), default=0)

    @memoize
    def pi(s):
        return max(actions(s), key=lambda a: Q(s, a))

    return Q, V, pi, info