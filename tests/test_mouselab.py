import pytest

from mouselab.cost_functions import distance_graph_cost
from mouselab.distributions import Categorical
from mouselab.graph_utils import get_structure_properties
from mouselab.mouselab import MouselabEnv
from mouselab.envs.registry import register

import numpy as np

structure = {
    "layout" : {
        "0" : [0, 0],
        "1" : [0, -1],
        "2" : [0, -2],
        "3" : [1, -2],
        "4" : [-1, -2],
        "5" : [1, 0],
        "6" : [2, 0],
        "7" : [2, -1],
        "8" : [2, 1],
        "9" : [-1, 0],
        "10": [-2, 0],
        "11": [-2, -1],
        "12": [-2, 1],
    },
    "initial": "0",
    "graph"  : {
        "0" : {"up": [0, "1"], "right": [0, "5"], "left": [0, "9"]},
        "1" : {"up": [0, "2"]},
        "2" : {"right": [0, "3"], "left": [0, "4"]},
        "3" : {},
        "4" : {},
        "5" : {"right": [0, "6"]},
        "6" : {"up": [0, "7"], "down": [0, "8"]},
        "7" : {},
        "8" : {},
        "9" : {"left": [0, "10"]},
        "10": {"up": [0, "11"], "down": [0, "12"]},
        "11": {},
        "12": {},
    },
}

depth_calculation_test_cases = [
    [
        [3, 1, 2],
        {
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 3,
            5 : 1,
            6 : 2,
            7 : 3,
            8 : 3,
            9 : 1,
            10: 2,
            11: 3,
            12: 3,
        },
        get_structure_properties(structure),
    ],
    [
        [3, 1, 1, 1, 2],
        {
            0 : 0,
            1 : 1,
            2 : 2,
            3 : 3,
            4 : 4,
            5 : 5,
            6 : 5,
            7 : 1,
            8 : 2,
            9 : 3,
            10: 4,
            11: 5,
            12: 5,
            13: 1,
            14: 2,
            15: 3,
            16: 4,
            17: 5,
            18: 5,
        },
        {},
    ],
]


@pytest.mark.parametrize(
    "branching,true_depths,mdp_graph_properties", depth_calculation_test_cases
)
def test_depth_calculation(branching, true_depths, mdp_graph_properties):
    env = MouselabEnv.new_symmetric(
        branching, 1, mdp_graph_properties=mdp_graph_properties
    )
    depth_dict = {node: data["depth"] for node, data in env.mdp_graph.nodes(data=True)}
    assert depth_dict == true_depths


@pytest.mark.parametrize("setting,expected", [["toy_imbalanced", False], ["high_increasing", True]])
def test_nonsymmetric(setting, expected):
    """
    Test to see if symmetric rewards on tree
    """

    env = MouselabEnv.new_registered(setting)

    symmetric = True
    depth_values = {}
    for node in env.mdp_graph.nodes:
        if env.mdp_graph.nodes[node]['depth'] not in depth_values:
            depth_values[env.mdp_graph.nodes[node]['depth']] = hash(env.init[node])
        else:
            if depth_values[env.mdp_graph.nodes[node]['depth']] != hash(env.init[node]):
                symmetric = False

    assert symmetric == expected

@pytest.mark.parametrize("power_utility", [0.1, .5 , 1])
def test_power_utility(power_utility):
    """
    Test power utility function
    """

    env = MouselabEnv.new_registered("high_increasing")
    env_with_power_utility = MouselabEnv.new_registered("high_increasing", power_utility=power_utility, ground_truth=env.ground_truth)

    # 4 random actions
    for action in range(4):
        possible_actions = list(env.actions(env._state))
        possible_actions.remove(env.term_action)

        curr_action = np.random.choice(possible_actions)
        print(curr_action)

        if env.expected_term_reward(env._state) >= 0:
            assert env.expected_term_reward(env._state)**(power_utility) == env_with_power_utility.expected_term_reward(env_with_power_utility._state)
        else:
            assert - abs(env.expected_term_reward(env._state)) ** (
                power_utility) == env_with_power_utility.expected_term_reward(env_with_power_utility._state)

        env.step(curr_action)
        env_with_power_utility.step(curr_action)

    _, _, reward, _ = env.step(env.term_action)
    _, _, reward_power_utility, _ = env_with_power_utility.step(env.term_action)

    if reward >= 0:
        assert reward**(power_utility) == reward_power_utility
    else:
        assert - abs(reward) ** (power_utility) == reward_power_utility
