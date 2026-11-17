import pytest
from autopark import ParkingEnvironment, DIRECTIONS, load_map_file


def test_default_map_loading():
    """Verify that default_map.txt loads correctly with expected dimensions."""
    env = ParkingEnvironment()
    assert env.height == 8
    assert env.width == 16
    assert len(env.layout) == 8
    assert all(len(row) == 16 for row in env.layout)


def test_multiple_targets_detected():
    """Verify that all 'P' parking stalls are identified in targets and target_set."""
    env = ParkingEnvironment()
    assert len(env.targets) >= 2, f"Expected multiple 'P' targets, found {len(env.targets)}"
    assert set(env.targets) == env.target_set
    # Backwards compatibility property
    assert env.target == env.targets[0]
    # Targets must be distinct from obstacles and start
    assert env.start not in env.target_set
    for t in env.targets:
        assert t not in env.obstacles
        assert t in env.passable


def test_start_gate_detected():
    """Verify entrance gate 'S' is correctly parsed."""
    env = ParkingEnvironment()
    assert env.start == (1, 6)
    assert env.start in env.passable
    assert env.start not in env.obstacles


def test_get_closest_target():
    """Verify Manhattan distance calculation to the nearest target bay."""
    env = ParkingEnvironment()
    # From entrance (1, 6), closest bay is (4, 1)
    closest = env.get_closest_target(1, 6)
    assert closest == (4, 1)

    # Near the far right, closest should be (12, 1)
    closest_right = env.get_closest_target(13, 3)
    assert closest_right == (12, 1)


def test_obstacles_detection():
    """Verify boundaries and obstacles are marked as obstacles."""
    env = ParkingEnvironment()
    # Top-left corner wall
    assert env.is_obstacle(0, 0) is True
    # Out of bounds
    assert env.is_obstacle(-1, 0) is True
    assert env.is_obstacle(0, -1) is True
    assert env.is_obstacle(16, 8) is True
    # Open driving lane (1, 2)
    assert env.is_obstacle(1, 2) is False


def test_lidar_sensor_readings():
    """Verify 4-directional raycasting proximity sensors."""
    env = ParkingEnvironment()
    # From entrance (1, 6) facing East (heading=1)
    # Front is East, Rear is West, Left is North, Right is South
    lidar = env.get_lidar(1, 6, heading=1)
    assert "front" in lidar
    assert "rear" in lidar
    assert "left" in lidar
    assert "right" in lidar
    assert lidar["rear"] == 0   # Immediately adjacent to left wall at x=0
    assert lidar["right"] == 0  # Immediately adjacent to bottom wall at y=7
    assert lidar["front"] > 0   # Open hallway to the east
    assert lidar["left"] > 0    # Open hallway to the north


def test_step_transitions():
    """Verify kinematic state updates for the 4 discrete actions."""
    env = ParkingEnvironment()
    start_state = (1, 4, 1) # (x=1, y=4, heading=East)

    # 0: FORWARD -> moves East to (2, 4)
    next_s, _, done, info = env.step(start_state, action=0)
    assert next_s == (2, 4, 1)
    assert not done
    assert info == "DRIVING"

    # 1: REVERSE -> backs up West to (0, 4) which is a wall!
    next_s_rev, reward, done_rev, info_rev = env.step(start_state, action=1)
    assert done_rev is True
    assert info_rev == "CRASH"
    assert reward == -60.0

    # 2: TURN_LEFT -> rotates counter-clockwise (East -> North = 0)
    next_s_tl, _, done_tl, _ = env.step(start_state, action=2)
    assert next_s_tl == (1, 4, 0)
    assert not done_tl

    # 3: TURN_RIGHT -> rotates clockwise (East -> South = 2)
    next_s_tr, _, done_tr, _ = env.step(start_state, action=3)
    assert next_s_tr == (1, 4, 2)
    assert not done_tr


def test_parking_in_any_target_succeeds():
    """Verify car successfully parks when entering ANY of the target 'P' stalls."""
    env = ParkingEnvironment()

    for target_pos in env.targets:
        tx, ty = target_pos
        # Position car immediately in front of the stall facing North (heading=0)
        approach_state = (tx, ty + 1, 0)
        next_state, reward, done, info = env.step(approach_state, action=0) # FORWARD into bay

        assert next_state == (tx, ty, 0)
        assert done is True
        assert info == "PARKED"
        assert reward == 280.0 # Maximum head-in bonus


def test_invalid_map_exceptions():
    """Verify ValueError is raised if a map lacks targets or a start gate."""
    no_target_map = [
        "#####",
        "#S  #",
        "#####"
    ]
    with pytest.raises(ValueError, match="at least one parking target"):
        ParkingEnvironment(layout=no_target_map)

    no_start_map = [
        "#####",
        "#P  #",
        "#####"
    ]
    with pytest.raises(ValueError, match="start gate"):
        ParkingEnvironment(layout=no_start_map)
