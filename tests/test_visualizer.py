from autopark import ParkingEnvironment, AsciiVisualizer, Color


def test_visualizer_without_color():
    """Verify that use_color=False outputs plain text without ANSI escape sequences."""
    env = ParkingEnvironment()
    vis = AsciiVisualizer(env, use_color=False)
    output = vis.c(Color.RED, "Hello World")
    assert output == "Hello World"
    assert "\033[" not in output


def test_sensor_bar_format():
    """Verify LIDAR gauge formatting."""
    env = ParkingEnvironment()
    vis = AsciiVisualizer(env, use_color=False)
    bar_2m = vis.render_sensor_bar(2, max_dist=8)
    assert "2m" in bar_2m
    assert "██░░░░░░" in bar_2m

    bar_8m = vis.render_sensor_bar(8, max_dist=8)
    assert "8m" in bar_8m
    assert "████████" in bar_8m


def test_render_frame_output():
    """Verify render_frame produces complete ASCII HUD with all targets visible."""
    env = ParkingEnvironment()
    vis = AsciiVisualizer(env, use_color=False)
    # Position car at (2, 4) away from start gate (1, 6)
    frame = vis.render_frame(state=(2, 4, 1), last_action=0, step_count=1, status="DRIVING")

    assert "AUTONOMOUS SELF-PARKING CAR" in frame
    assert "Front Radar:" in frame
    assert "Rear Radar :" in frame
    assert "Target :" in frame
    # Verify all targets 'P' are rendered on the map (+1 in the legend at the bottom)
    assert frame.count("[P]") == len(env.targets) + 1
    # Verify starting gate is rendered
    assert "[S]" in frame
    # Verify car symbol is rendered
    assert "[>]" in frame
