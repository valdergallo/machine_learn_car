import pytest
from autopark import ParkingEnvironment, QLearningAgent, train_agent


def test_q_table_default():
    """Verify that unvisited states initialize with 4 zero Q-values."""
    agent = QLearningAgent()
    unseen_state = (3, 5, 2)
    q_vals = agent.get_q(unseen_state)
    assert q_vals == [0.0, 0.0, 0.0, 0.0]


def test_action_selection():
    """Verify greedy action selection (explore=False) chooses the maximum Q-value."""
    agent = QLearningAgent(epsilon=0.0)
    state = (2, 2, 0)
    # Artificially set Q-values
    agent.q_table[state] = [5.0, 2.0, 10.5, -1.0]
    # Action 2 (TURN_LEFT) has the highest value (10.5)
    best_action = agent.select_action(state, explore=False)
    assert best_action == 2


def test_bellman_update():
    """Verify the TD-error update rule: Q(s, a) += alpha * (r + gamma * max(Q(s')) - Q(s, a))."""
    agent = QLearningAgent(alpha=0.5, gamma=0.9)
    state = (1, 1, 0)
    next_state = (1, 0, 0)
    action = 0
    reward = 10.0

    # Next state has max Q of 20.0
    agent.q_table[next_state] = [0.0, 20.0, 0.0, 0.0]

    # Target = 10.0 + 0.9 * 20.0 = 28.0
    # Update = 0.0 + 0.5 * (28.0 - 0.0) = 14.0
    agent.update(state, action, reward, next_state, done=False)
    assert pytest.approx(agent.get_q(state)[action], 0.01) == 14.0


def test_epsilon_decay():
    """Verify exploration factor decays down to eps_min."""
    agent = QLearningAgent(epsilon=1.0, eps_min=0.05, eps_decay=0.5)
    agent.decay_epsilon()
    assert agent.epsilon == 0.5
    agent.decay_epsilon()
    assert agent.epsilon == 0.25
    # Decay multiple times to hit floor
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == 0.05


def test_short_training_run():
    """Verify train_agent runs without exceptions and updates Q-table."""
    env = ParkingEnvironment()
    agent = QLearningAgent(alpha=0.3, gamma=0.95, epsilon=1.0)
    trained_agent = train_agent(env, agent=agent, episodes=40, verbose=False)
    assert len(trained_agent.q_table) > 0
    assert trained_agent.epsilon < 1.0
