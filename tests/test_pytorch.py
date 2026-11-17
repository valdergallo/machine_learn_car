import tempfile
import os
import pytest

torch = pytest.importorskip("torch")
from autopark import ParkingEnvironment
from autopark_pytorch import QNetwork, ReplayBuffer, DQNAgent, train_dqn


def test_q_network_forward_pass():
    """Verify QNetwork output tensor shape for batches and single inputs."""
    net = QNetwork(state_dim=12, action_dim=4)
    # Single sample
    x = torch.randn(12)
    out = net(x)
    assert out.shape == (4,)

    # Batch of 16
    batch_x = torch.randn(16, 12)
    batch_out = net(batch_x)
    assert batch_out.shape == (16, 4)


def test_replay_buffer_operations():
    """Verify pushing experiences and drawing mini-batches from ReplayBuffer."""
    buffer = ReplayBuffer(capacity=10)
    assert len(buffer) == 0

    s = torch.randn(12)
    s_next = torch.randn(12)

    for i in range(8):
        buffer.push(s, action=1, reward=1.0, next_state_vec=s_next, done=False)

    assert len(buffer) == 8
    states, actions, rewards, next_states, dones = buffer.sample(batch_size=4)
    assert states.shape == (4, 12)
    assert actions.shape == (4,)
    assert rewards.shape == (4,)
    assert next_states.shape == (4, 12)
    assert dones.shape == (4,)


def test_dqn_agent_state_encoding():
    """Verify 12-dimensional continuous state vector encoding."""
    env = ParkingEnvironment()
    agent = DQNAgent(env)
    vec = agent.get_state_vector((1, 6, 1))

    assert isinstance(vec, torch.Tensor)
    assert vec.shape == (12,)
    # Normalized position within [0, 1]
    assert 0.0 <= vec[0].item() <= 1.0
    assert 0.0 <= vec[1].item() <= 1.0


def test_dqn_agent_get_q():
    """Verify get_q returns 4 float values for visualizer HUD."""
    env = ParkingEnvironment()
    agent = DQNAgent(env)
    q_vals = agent.get_q((1, 6, 1))

    assert isinstance(q_vals, list)
    assert len(q_vals) == 4
    assert all(isinstance(v, float) for v in q_vals)


def test_dqn_checkpoint_save_and_load():
    """Verify saving weights to file and loading them back."""
    env = ParkingEnvironment()
    agent = DQNAgent(env)

    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        agent.save(tmp_path)
        assert os.path.exists(tmp_path)

        new_agent = DQNAgent(env)
        loaded = new_agent.load(tmp_path)
        assert loaded is True

        # Verify weights match
        for p1, p2 in zip(agent.policy_net.parameters(), new_agent.policy_net.parameters()):
            assert torch.equal(p1, p2)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_short_dqn_training_run():
    """Verify train_dqn completes a short training loop without exceptions."""
    env = ParkingEnvironment()
    agent = DQNAgent(env)
    trained = train_dqn(env, agent, episodes=10, max_steps=15, verbose=False)
    assert trained.epsilon < 1.0
