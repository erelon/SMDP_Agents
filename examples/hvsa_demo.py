"""Harmonic against Q-Learning on the two-path chain, one transition at a time.

The point of :mod:`examples.envs.two_path` is that path A totals 101 and path B
totals 100, but path B pays 50 of its 100 immediately while path A pays 1. This
script walks both agents through the two paths by hand — forcing the first
decision so that each of them is visited — printing rho before and after every
update, and then the Q-table and the greedy choice each agent ends up with.

Everything here is deliberately hand-driven and tiny: with two decisions, four
transitions and unit holding times, you can check the update arithmetic by eye,
which is what makes this the fixture to reach for when a rate estimator starts
behaving oddly.

Run with ``python -m examples.hvsa_demo``. Moved from ``hvsa.py`` at the
repository root.
"""

from agents import Harmonic, QLearning

from .envs.two_path import ACTION_A, ACTION_B, STATE0, TwoPathEnv

#: The first decision is forced to each of the two actions in turn, so both paths
#: are visited regardless of what the agent would have explored.
FORCED_FIRST_DECISIONS = (ACTION_B, ACTION_A)
EPISODES = 2


def walk(agent, env, episodes=EPISODES, verbose=True):
    """Drive ``agent`` through both paths ``episodes`` times; return its choice."""
    for episode in range(episodes):
        state, _ = env.reset()
        agent.initialize_table(state)

        for forced in FORCED_FIRST_DECISIONS:
            state, _ = env.reset()
            if verbose:
                print(f"  episode {episode} forcing "
                      f"{'A' if forced == ACTION_A else 'B'}: rho={agent.rho:.6g}", end="")

            next_state, reward, _, _, info = env.step(forced)
            agent.learn(state, forced, reward, next_state, info["tau"])
            if verbose:
                print(f" -> {agent.rho:.6g}", end="")

            # The second decision is forced by the environment anyway, but let
            # the agent pick it so its exploration is exercised.
            action = agent.act(next_state)
            final_state, reward, _, _, info = env.step(action)
            agent.learn(next_state, action, reward, final_state, info["tau"])
            if verbose:
                print(f" -> {agent.rho:.6g}")

    if verbose:
        print(f"  q_table: {agent.q_table}")
    return agent.eval(env.reset()[0])


def main():
    env = TwoPathEnv()
    agents = [
        Harmonic("Harmonic", [ACTION_A, ACTION_B], env=env, exploration_rate=1,
                 rho_learning_rate=1, seed=1, learning_rate=1, with_rho_trick=False),
        QLearning("Qlearner", [ACTION_A, ACTION_B], env=env, exploration_rate=1,
                  learning_rate=1, seed=1, discount_factor=1),
    ]
    for agent in agents:
        print(f"{agent.name}:")
        choice = walk(agent, env)
        print(f"  chose action {'A' if choice == ACTION_A else 'B'} in state0 "
              f"({'patient, total 101' if choice == ACTION_A else 'greedy, total 100'})")
        print("#" * 60)


if __name__ == "__main__":
    main()
