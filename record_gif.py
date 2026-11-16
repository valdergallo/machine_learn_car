#!/usr/bin/env python3
"""
GIF Recorder & HTML Player Generator for Autonomous Parking Simulation
Captures the ASCII simulation frames and exports them as an animated GIF and/or interactive HTML player.
"""

import os
import sys
import time
import argparse
import re
import unicodedata

# Import the parking simulation engine
import autopark

# -----------------------------------------------------------------------------
# Color Definitions for Image & HTML Rendering
# -----------------------------------------------------------------------------
ANSI_COLOR_MAP = {
    '30': '#4e4e4e', '90': '#6c7086', # Gray / Dark
    '31': '#f38ba8', '91': '#f38ba8', # Red
    '32': '#a6e3a1', '92': '#a6e3a1', # Green
    '33': '#f9e2af', '93': '#f9e2af', # Yellow
    '34': '#89b4fa', '94': '#89b4fa', # Blue
    '35': '#cba6f7', '95': '#cba6f7', # Magenta
    '36': '#89dceb', '96': '#89dceb', # Cyan
    '37': '#cdd6f4', '97': '#ffffff', # White
}

def parse_ansi_line(line):
    """Parses a string containing ANSI color codes into (text, fg_color, bg_color) tokens."""
    tokens = []
    pattern = re.compile(r'\033\[([0-9;]+)m')
    current_fg = '#cdd6f4' # Default foreground
    current_bg = None      # Default background
    last_end = 0
    
    for match in pattern.finditer(line):
        text = line[last_end:match.start()]
        if text:
            tokens.append((text, current_fg, current_bg))
        codes = match.group(1).split(';')
        for c in codes:
            if c == '0':
                current_fg = '#cdd6f4'
                current_bg = None
            elif c in ANSI_COLOR_MAP:
                current_fg = ANSI_COLOR_MAP[c]
            elif c == '42': # Green background (PARKED)
                current_bg = '#1e6600'
            elif c == '41': # Red background (CRASH)
                current_bg = '#8b0000'
        last_end = match.end()
        
    tail = line[last_end:]
    if tail:
        tokens.append((tail, current_fg, current_bg))
    return tokens


# -----------------------------------------------------------------------------
# Frame Collector: Runs simulation and captures text frames
# -----------------------------------------------------------------------------
def capture_simulation_frames(env, agent, start_state=None, max_steps=50):
    if start_state is None:
        start_state = (env.start[0], env.start[1], 1) # Entrance facing East
        
    visualizer = autopark.AsciiVisualizer(env, use_color=True)
    state = start_state
    visited = set()
    total_reward = 0.0
    status = "DRIVING"
    last_action = None
    
    frames = []
    
    for step in range(max_steps):
        visited.add((state[0], state[1]))
        frame_text = visualizer.render_frame(
            state=state,
            last_action=last_action,
            step_count=step,
            total_reward=total_reward,
            status=status,
            visited_path=visited,
            q_agent=agent
        )
        frames.append(frame_text)
        
        if status in ("PARKED", "CRASH"):
            # Hold the final parked frame for 5 extra frames so it pauses nicely
            for _ in range(5):
                frames.append(frame_text)
            break
            
        action = agent.select_action(state, explore=False)
        last_action = action
        next_state, reward, done, info = env.step(state, action)
        total_reward += reward
        status = info
        state = next_state
        
    return frames


# -----------------------------------------------------------------------------
# GIF Exporter (Requires Pillow)
# -----------------------------------------------------------------------------
def export_gif(frames, output_path="autopark.gif", fps=5, font_size=15):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("\n\033[93m[!] Pillow is not installed in this Python environment.\033[0m")
        print("To generate animated .GIF files directly with Python, install Pillow:")
        print("    \033[96mpip install pillow\033[0m\n")
        return False
        
    print(f"[*] Rendering {len(frames)} frames into animated GIF using Pillow...")
    
    # Try finding a crisp monospace font
    font = None
    font_candidates = [
        "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/Library/Fonts/Courier New.ttf",
        "C:\\Windows\\Fonts\\consola.ttf",
        "DejaVuSansMono.ttf"
    ]
    for candidate in font_candidates:
        if os.path.exists(candidate):
            try:
                font = ImageFont.truetype(candidate, font_size)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    # Determine canvas dimensions based on frame 0
    sample_lines = frames[0].split('\n')
    max_cols = max(autopark.visible_width(l) for l in sample_lines)
    num_rows = len(sample_lines)
    
    char_w = font_size * 0.62
    char_h = font_size * 1.35
    
    padding_x = 24
    padding_y = 20
    img_w = int(max_cols * char_w + padding_x * 2)
    img_h = int(num_rows * char_h + padding_y * 2)
    
    bg_color = (24, 24, 37) # Dark terminal background
    
    pil_images = []
    for frame_idx, frame_text in enumerate(frames):
        img = Image.new("RGB", (img_w, img_h), color=bg_color)
        draw = ImageDraw.Draw(img)
        
        lines = frame_text.split('\n')
        curr_y = padding_y
        
        for line in lines:
            tokens = parse_ansi_line(line)
            curr_x = padding_x
            
            for text, fg_hex, bg_hex in tokens:
                for ch in text:
                    w = 2 if unicodedata.east_asian_width(ch) in ('W', 'F') else 1
                    ch_pixel_w = char_w * w
                    
                    if bg_hex:
                        draw.rectangle(
                            [curr_x, curr_y, curr_x + ch_pixel_w, curr_y + char_h],
                            fill=bg_hex
                        )
                    draw.text((curr_x, curr_y), ch, font=font, fill=fg_hex)
                    curr_x += ch_pixel_w
                    
            curr_y += char_h
            
        pil_images.append(img)
        
    duration_ms = int(1000 / fps)
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration_ms,
        loop=0
    )
    print(f"\033[92m[✔] Animated GIF saved successfully to: {output_path} ({len(frames)} frames)\033[0m")
    return True


# -----------------------------------------------------------------------------
# Standalone HTML Player Generator (100% Zero-Dependency Fallback)
# -----------------------------------------------------------------------------
def export_html_player(frames, output_path="autopark_player.html", fps=5):
    """Exports a self-contained interactive web player with retro CRT terminal styling."""
    escaped_frames = []
    for f in frames:
        # Convert ANSI to HTML spans
        lines = f.split('\n')
        html_lines = []
        for line in lines:
            tokens = parse_ansi_line(line)
            spans = []
            for text, fg, bg in tokens:
                safe_text = (text.replace('&', '&amp;')
                                 .replace('<', '&lt;')
                                 .replace('>', '&gt;')
                                 .replace(' ', '&nbsp;'))
                bg_style = f"background-color: {bg};" if bg else ""
                spans.append(f'<span style="color: {fg}; {bg_style}">{safe_text}</span>')
            html_lines.append("".join(spans))
        escaped_frames.append("<br>".join(html_lines))

    frames_json = str(escaped_frames)
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>🚗 Autonomous Self-Parking Car - ASCII Animation</title>
<style>
  body {{
    background-color: #0f111a;
    color: #cdd6f4;
    font-family: ui-monospace, 'Cascadia Code', 'Ubuntu Mono', 'Courier New', monospace;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 100vh;
    margin: 0;
  }}
  .terminal {{
    background-color: #181825;
    border: 2px solid #313244;
    border-radius: 12px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.7);
    padding: 24px;
    font-size: 15px;
    line-height: 1.35;
    white-space: nowrap;
    user-select: none;
  }}
  .controls {{
    margin-top: 18px;
    display: flex;
    gap: 12px;
    align-items: center;
  }}
  button {{
    background: #89b4fa;
    color: #11111b;
    border: none;
    padding: 8px 18px;
    font-size: 14px;
    font-weight: bold;
    border-radius: 6px;
    cursor: pointer;
    transition: 0.15s ease;
  }}
  button:hover {{
    background: #b4befe;
  }}
  .info {{
    margin-top: 10px;
    color: #6c7086;
    font-size: 13px;
  }}
</style>
</head>
<body>
  <div class="terminal" id="display"></div>
  <div class="controls">
    <button onclick="togglePlay()" id="playBtn">Pause</button>
    <button onclick="restart()">Restart</button>
    <label style="color:#a6adc8; font-size:14px;">Speed:
      <input type="range" id="speed" min="1" max="15" value="{fps}" onchange="updateSpeed(this.value)">
    </label>
  </div>
  <div class="info">Frame: <span id="frameNum">1</span> / {len(frames)}</div>

<script>
  const frames = {frames_json};
  let current = 0;
  let isPlaying = true;
  let fps = {fps};
  let timer = null;

  function render() {{
    document.getElementById('display').innerHTML = frames[current];
    document.getElementById('frameNum').innerText = current + 1;
  }}

  function nextFrame() {{
    current = (current + 1) % frames.length;
    render();
  }}

  function schedule() {{
    clearInterval(timer);
    if (isPlaying) {{
      timer = setInterval(nextFrame, 1000 / fps);
    }}
  }}

  function togglePlay() {{
    isPlaying = !isPlaying;
    document.getElementById('playBtn').innerText = isPlaying ? "Pause" : "Play";
    schedule();
  }}

  function restart() {{
    current = 0;
    render();
  }}

  function updateSpeed(newFps) {{
    fps = newFps;
    schedule();
  }}

  render();
  schedule();
</script>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\033[92m[✔] Interactive HTML Web Player saved to: {output_path}\033[0m")
    return True


# -----------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Record Autonomous Parking Car Simulation to Animated GIF / HTML")
    parser.add_argument("--output", default="autopark.gif", help="Output GIF file path (default: autopark.gif)")
    parser.add_argument("--html", default="autopark_player.html", help="Output HTML player path (default: autopark_player.html)")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second in GIF/player (default: 5)")
    parser.add_argument("--font-size", type=int, default=15, help="Monospace font size in GIF (default: 15)")
    args = parser.parse_args()
    
    print("\033[96m╔════════════════════════════════════════════════════════════════╗\033[0m")
    print("\033[96m║\033[0m \033[1m\033[97m🚗 RECORDING AUTONOMOUS PARKING RUN TO ANIMATED GIF / HTML    \033[0m\033[96m║\033[0m")
    print("\033[96m╚════════════════════════════════════════════════════════════════╝\033[0m")
    
    env = autopark.ParkingEnvironment()
    print("[*] Training AI agent for simulation run...")
    agent = autopark.train_agent(env, episodes=2500, verbose=False)
    
    print("[*] Capturing autonomous parking trajectory frames...")
    frames = capture_simulation_frames(env, agent)
    print(f"[*] Captured {len(frames)} simulation frames.")
    
    # 1. Export HTML Web Player (Always works 100%, zero-dependency)
    export_html_player(frames, output_path=args.html, fps=args.fps)
    
    # 2. Export GIF (Uses Pillow if available)
    gif_ok = export_gif(frames, output_path=args.output, fps=args.fps, font_size=args.font_size)
    
    print("\n\033[1mDone!\033[0m You can view \033[96m" + args.html + "\033[0m in any browser, or use the GIF methods below.")

if __name__ == "__main__":
    main()
