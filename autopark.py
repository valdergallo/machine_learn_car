#!/usr/bin/env python3
"""
Autonomous Self-Parking Car Simulation with Q-Learning and ASCII Graphics
Built with standard Python library (no external dependencies required).
"""

import sys
import os
import time
import random
import argparse
import unicodedata
import re

# -----------------------------------------------------------------------------
# ANSI Color Codes & String Width Helpers
# -----------------------------------------------------------------------------
class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"
    BG_GREEN = "\033[42m"
    BG_RED   = "\033[41m"
    BG_BLUE  = "\033[44m"

ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def char_width(c):
    return 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1

def visible_width(s):
    clean = ANSI_RE.sub('', s)
    return sum(char_width(c) for c in clean)

def pad_to(s, target_w):
    w = visible_width(s)
    return s + " " * max(0, target_w - w)

# Direction vectors: 0: North, 1: East, 2: South, 3: West
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
DIR_NAMES  = ["NORTH", "EAST", "SOUTH", "WEST"]
DIR_ARROWS = ["▲", "►", "▼", "◄"]

ACTION_NAMES = ["FORWARD", "REVERSE", "TURN_LEFT", "TURN_RIGHT"]
ACTION_SYMBOLS = ["▲ FWD", "▼ REV", "◄ LEFT", "► RIGHT"]

# -----------------------------------------------------------------------------
# Parking Environment
# -----------------------------------------------------------------------------
DEFAULT_MAP = [
    "################",  # 0 Wall
    "#|C| |C|P|C| | #",  # 1 Top bays: Parked Cars 'C', Target 'P' at (8, 1)
    "#              #",  # 2 Upper driveway / maneuvering aisle
    "#              #",  # 3 Driving lane
    "#              #",  # 4 Driving lane
    "#   |C| |C| |C|#",  # 5 Lower parked cars island
    "#S             #",  # 6 Entrance / Start area (1, 6)
    "################",  # 7 Bottom Wall
]

class ParkingEnvironment:
    def __init__(self, layout=None):
        self.layout = layout or DEFAULT_MAP
        self.height = len(self.layout)
        self.width  = len(self.layout[0])
        
        self.obstacles = set()
        self.passable  = set()
        self.target    = None
        self.start     = None
        
        for y in range(self.height):
            for x in range(self.width):
                ch = self.layout[y][x]
                if ch in ('#', 'C', '|'):
                    self.obstacles.add((x, y))
                elif ch == 'P':
                    self.target = (x, y)
                    self.passable.add((x, y))
                elif ch == 'S':
                    self.start = (x, y)
                    self.passable.add((x, y))
                else:
                    self.passable.add((x, y))
                    
        self.valid_spawns = [p for p in self.passable if p != self.target]
        
    def is_obstacle(self, x, y):
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return (x, y) in self.obstacles

    def get_lidar(self, x, y, heading):
        """Raycast proximity sensors: Front, Rear, Left, Right distances."""
        sensor_dirs = {
            "front": heading,
            "rear":  (heading + 2) % 4,
            "left":  (heading - 1) % 4,
            "right": (heading + 1) % 4
        }
        readings = {}
        for name, d_idx in sensor_dirs.items():
            dx, dy = DIRECTIONS[d_idx]
            dist = 0
            cx, cy = x, y
            while True:
                cx += dx
                cy += dy
                if self.is_obstacle(cx, cy):
                    break
                dist += 1
            readings[name] = dist
        return readings

    def step(self, state, action):
        """
        Execute an action:
        0: FORWARD, 1: REVERSE, 2: TURN_LEFT, 3: TURN_RIGHT
        Returns: (next_state, reward, done, info)
        """
        x, y, o = state
        
        if action == 0:    # FORWARD
            nx, ny, no = x + DIRECTIONS[o][0], y + DIRECTIONS[o][1], o
        elif action == 1:  # REVERSE
            nx, ny, no = x - DIRECTIONS[o][0], y - DIRECTIONS[o][1], o
        elif action == 2:  # TURN_LEFT
            nx, ny, no = x, y, (o - 1) % 4
        elif action == 3:  # TURN_RIGHT
            nx, ny, no = x, y, (o + 1) % 4
        else:
            raise ValueError(f"Invalid action {action}")
            
        if self.is_obstacle(nx, ny):
            # Collision penalty
            return state, -60.0, True, "CRASH"
            
        if (nx, ny) == self.target:
            # Parked! Extra bonus if parked head-in facing North (0)
            bonus = 280.0 if no == 0 else 220.0
            return (nx, ny, no), bonus, True, "PARKED"
            
        # Transit step: step penalty + distance shaping toward target
        d_curr = abs(x - self.target[0]) + abs(y - self.target[1])
        d_next = abs(nx - self.target[0]) + abs(ny - self.target[1])
        
        # Reward forward road driving over reversing in open lane
        rev_penalty = -0.3 if action == 1 else 0.0
        reward = -1.0 + 0.6 * (d_curr - d_next) + rev_penalty
        
        return (nx, ny, no), reward, False, "DRIVING"


# -----------------------------------------------------------------------------
# Q-Learning Agent
# -----------------------------------------------------------------------------
class QLearningAgent:
    def __init__(self, alpha=0.2, gamma=0.98, epsilon=1.0, eps_min=0.02, eps_decay=0.999):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        
    def get_q(self, state):
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0, 0.0, 0.0]
        return self.q_table[state]

    def select_action(self, state, explore=True):
        if explore and random.random() < self.epsilon:
            return random.randint(0, 3)
        q_vals = self.get_q(state)
        max_val = max(q_vals)
        best_actions = [i for i, v in enumerate(q_vals) if v == max_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, done):
        current_q = self.get_q(state)[action]
        if done:
            target = reward
        else:
            best_future = max(self.get_q(next_state))
            target = reward + self.gamma * best_future
            
        self.q_table[state][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# -----------------------------------------------------------------------------
# ASCII Visualizer & HUD
# -----------------------------------------------------------------------------
class AsciiVisualizer:
    def __init__(self, env, use_color=True):
        self.env = env
        self.use_color = use_color
        
        # Car representation by orientation
        self.car_chars = {
            0: "[^]", # North
            1: "[>]", # East
            2: "[v]", # South
            3: "[<]"  # West
        }
        
    def c(self, color_code, text):
        if not self.use_color:
            return text
        return f"{color_code}{text}{Color.RESET}"

    def render_sensor_bar(self, distance, max_dist=8):
        filled = min(distance, max_dist)
        empty  = max(0, max_dist - filled)
        bar = "█" * filled + "░" * empty
        return f"[{self.c(Color.GREEN if filled > 2 else Color.RED, bar)}] {distance:2d}m"

    def render_frame(self, state, last_action=None, step_count=0, 
                     total_reward=0.0, status="DRIVING", 
                     visited_path=None, q_agent=None, ep_num=None):
        x, y, o = state
        lidar = self.env.get_lidar(x, y, o)
        dist_to_target = abs(x - self.env.target[0]) + abs(y - self.env.target[1])
        q_vals = q_agent.get_q(state) if q_agent else [0.0, 0.0, 0.0, 0.0]
        
        W = 64
        w1, w2 = 31, 32
        
        lines = []
        lines.append(self.c(Color.CYAN, "┌" + "─" * W + "┐"))
        title_str = "        🚗 AUTONOMOUS SELF-PARKING CAR (REINFORCEMENT LEARNING)"
        lines.append(self.c(Color.CYAN, "│") + pad_to(self.c(Color.BOLD + Color.WHITE, title_str), W) + self.c(Color.CYAN, "│"))
        lines.append(self.c(Color.CYAN, "├" + "─" * w1 + "┬" + "─" * w2 + "┤"))
        
        # Telemetry & LIDAR (Split rows)
        act_str = ACTION_NAMES[last_action] if last_action is not None else "READY"
        status_col = Color.GREEN if status == "PARKED" else (Color.RED if status == "CRASH" else Color.YELLOW)
        
        r1_c1 = f" Status : {self.c(status_col + Color.BOLD, status.ljust(8))} ({act_str})"
        r1_c2 = f" Front Radar: {self.render_sensor_bar(lidar['front'])}"
        lines.append(self.c(Color.CYAN, "│") + pad_to(r1_c1, w1) + self.c(Color.CYAN, "│") + pad_to(r1_c2, w2) + self.c(Color.CYAN, "│"))
        
        heading_display = f"{DIR_NAMES[o]} {DIR_ARROWS[o]}"
        r2_c1 = f" Heading: {self.c(Color.WHITE + Color.BOLD, heading_display)}"
        r2_c2 = f" Rear Radar : {self.render_sensor_bar(lidar['rear'])}"
        lines.append(self.c(Color.CYAN, "│") + pad_to(r2_c1, w1) + self.c(Color.CYAN, "│") + pad_to(r2_c2, w2) + self.c(Color.CYAN, "│"))
        
        pos_display = f"(X:{x:2d}, Y:{y:2d})  Step:{step_count:2d}"
        r3_c1 = f" Pos    : {pos_display}"
        r3_c2 = f" Left Radar : {self.render_sensor_bar(lidar['left'])}"
        lines.append(self.c(Color.CYAN, "│") + pad_to(r3_c1, w1) + self.c(Color.CYAN, "│") + pad_to(r3_c2, w2) + self.c(Color.CYAN, "│"))
        
        tgt_display = f"Bay (8,1)  Dist: {dist_to_target}m"
        r4_c1 = f" Target : {self.c(Color.GREEN, tgt_display)}"
        r4_c2 = f" Right Radar: {self.render_sensor_bar(lidar['right'])}"
        lines.append(self.c(Color.CYAN, "│") + pad_to(r4_c1, w1) + self.c(Color.CYAN, "│") + pad_to(r4_c2, w2) + self.c(Color.CYAN, "│"))
        
        lines.append(self.c(Color.CYAN, "├" + "─" * W + "┤"))
        
        # Q-Values (AI Brain Confidence)
        best_a = q_vals.index(max(q_vals)) if q_vals else 0
        def fmt_q(idx):
            symbol = ACTION_SYMBOLS[idx]
            val = q_vals[idx]
            txt = f"{symbol}:{val:+6.1f}"
            if idx == best_a:
                return self.c(Color.GREEN + Color.BOLD, f"{txt}*")
            return self.c(Color.GRAY, f"{txt} ")

        q_line1 = f" AI Brain Q-Values:  {fmt_q(0)}   {fmt_q(2)}"
        q_line2 = f"                     {fmt_q(1)}   {fmt_q(3)}"
        lines.append(self.c(Color.CYAN, "│") + pad_to(q_line1, W) + self.c(Color.CYAN, "│"))
        lines.append(self.c(Color.CYAN, "│") + pad_to(q_line2, W) + self.c(Color.CYAN, "│"))
        
        lines.append(self.c(Color.CYAN, "├" + "─" * W + "┤"))
        lines.append(self.c(Color.CYAN, "│") + pad_to(" " + self.c(Color.WHITE + Color.BOLD, "PARKING LOT TOP-DOWN RADAR:"), W) + self.c(Color.CYAN, "│"))
        
        # Parking Lot Map Rendering
        visited = visited_path or set()
        for py in range(self.env.height):
            row_cells = []
            for px in range(self.env.width):
                if (px, py) == (x, y):
                    # Agent Car
                    car_sym = self.car_chars[o]
                    if status == "PARKED":
                        row_cells.append(self.c(Color.BG_GREEN + Color.WHITE + Color.BOLD, car_sym))
                    elif status == "CRASH":
                        row_cells.append(self.c(Color.BG_RED + Color.WHITE + Color.BOLD, "[*]"))
                    else:
                        row_cells.append(self.c(Color.CYAN + Color.BOLD, car_sym))
                elif (px, py) == self.env.target:
                    # Target Bay
                    row_cells.append(self.c(Color.GREEN + Color.BOLD, "[P]"))
                elif (px, py) == self.env.start:
                    # Start Gate
                    if (px, py) in visited:
                        row_cells.append(self.c(Color.YELLOW, " · "))
                    else:
                        row_cells.append(self.c(Color.BLUE + Color.BOLD, "[S]"))
                elif (px, py) in visited:
                    # Tire track trail
                    row_cells.append(self.c(Color.YELLOW, " · "))
                else:
                    ch = self.env.layout[py][px]
                    if ch == '#':
                        row_cells.append(self.c(Color.GRAY, "███"))
                    elif ch == 'C':
                        row_cells.append(self.c(Color.RED + Color.BOLD, "[X]"))
                    elif ch == '|':
                        row_cells.append(self.c(Color.WHITE, " | "))
                    else:
                        row_cells.append("   ")
                        
            map_row = "   " + "".join(row_cells)
            lines.append(self.c(Color.CYAN, "│") + pad_to(map_row, W) + self.c(Color.CYAN, "│"))
            
        lines.append(self.c(Color.CYAN, "├" + "─" * W + "┤"))
        legend = f" Legend: {self.c(Color.CYAN, '[^]')} Car  {self.c(Color.GREEN, '[P]')} Target  {self.c(Color.RED, '[X]')} Parked Car  {self.c(Color.YELLOW, '·')} Tire Track"
        lines.append(self.c(Color.CYAN, "│") + pad_to(legend, W) + self.c(Color.CYAN, "│"))
        lines.append(self.c(Color.CYAN, "└" + "─" * W + "┘"))
        
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Training and Simulation Engine
# -----------------------------------------------------------------------------
def train_agent(env, agent=None, episodes=3000, verbose=True):
    if agent is None:
        agent = QLearningAgent(alpha=0.2, gamma=0.98, epsilon=1.0, eps_min=0.02, eps_decay=0.998)
        
    start_time = time.time()
    successes = 0
    recent_history = []
    
    if verbose:
        print("\n" + Color.CYAN + "═" * 66 + Color.RESET)
        print(f" {Color.BOLD}🧠 TRAINING AI REINFORCEMENT LEARNING AGENT ({episodes:,} EPISODES)...{Color.RESET}")
        print(Color.CYAN + "═" * 66 + Color.RESET)
        
    for ep in range(1, episodes + 1):
        # 50% start at entrance gate, 50% from random open bay/aisle
        if ep % 2 == 0:
            sx, sy = env.start
            so = 1 # facing East
        else:
            sx, sy = random.choice(env.valid_spawns)
            so = random.randint(0, 3)
            
        state = (sx, sy, so)
        ep_reward = 0
        done = False
        
        for step in range(60):
            action = agent.select_action(state, explore=True)
            next_state, reward, done, info = env.step(state, action)
            agent.update(state, action, reward, next_state, done)
            ep_reward += reward
            state = next_state
            if done:
                if info == "PARKED":
                    recent_history.append(1)
                else:
                    recent_history.append(0)
                break
        else:
            recent_history.append(0)
            
        agent.decay_epsilon()
        
        if len(recent_history) > 100:
            recent_history.pop(0)
            
        if verbose and (ep % (episodes // 20) == 0 or ep == episodes):
            rate = (sum(recent_history) / len(recent_history)) * 100 if recent_history else 0
            bar_len = 24
            filled = int(bar_len * (ep / episodes))
            pbar = "█" * filled + "░" * (bar_len - filled)
            elapsed = time.time() - start_time
            sys.stdout.write(f"\r Progress: [{pbar}] {ep:4d}/{episodes} ep | Success Rate: {rate:5.1f}% | Time: {elapsed:.2f}s ")
            sys.stdout.flush()
            
    if verbose:
        print("\n" + Color.GREEN + " ✔ Model Training Complete! AI has mastered parking lot navigation." + Color.RESET)
        time.sleep(0.6)
        
    return agent


def run_animated_episode(env, agent, visualizer, start_state=None, 
                         step_delay=0.22, ep_num=None, title=None):
    if start_state is None:
        start_state = (env.start[0], env.start[1], 1) # Start at entrance facing East
        
    state = start_state
    visited = set()
    total_reward = 0.0
    status = "DRIVING"
    last_action = None
    
    # Hide terminal cursor for smooth animation
    sys.stdout.write("\033[?25l")
    try:
        for step in range(50):
            visited.add((state[0], state[1]))
            
            # Clear screen and draw frame
            frame = visualizer.render_frame(
                state=state,
                last_action=last_action,
                step_count=step,
                total_reward=total_reward,
                status=status,
                visited_path=visited,
                q_agent=agent,
                ep_num=ep_num
            )
            
            # Position cursor at top-left
            sys.stdout.write("\033[H\033[2J")
            if title:
                print(f" {Color.BOLD}{Color.WHITE}=== {title} ==={Color.RESET}\n")
            print(frame)
            sys.stdout.flush()
            
            if status in ("PARKED", "CRASH"):
                break
                
            time.sleep(step_delay)
            
            action = agent.select_action(state, explore=False)
            last_action = action
            next_state, reward, done, info = env.step(state, action)
            total_reward += reward
            status = info
            state = next_state
            
        time.sleep(0.8)
    finally:
        # Restore terminal cursor
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        
    return status == "PARKED"


def live_training_view(env, visualizer, episodes=400, step_delay=0.04):
    """Watch the car learn live, transitioning from random wandering to parking mastery!"""
    agent = QLearningAgent(alpha=0.25, gamma=0.98, epsilon=1.0, eps_min=0.03, eps_decay=0.992)
    
    sys.stdout.write("\033[?25l")
    try:
        for ep in range(1, episodes + 1):
            if ep % 2 == 0:
                sx, sy = env.start
                so = 1
            else:
                sx, sy = random.choice(env.valid_spawns)
                so = random.randint(0, 3)
                
            state = (sx, sy, so)
            visited = set()
            total_reward = 0
            
            for step in range(35):
                visited.add((state[0], state[1]))
                action = agent.select_action(state, explore=True)
                next_state, reward, done, info = env.step(state, action)
                agent.update(state, action, reward, next_state, done)
                total_reward += reward
                
                # Render every frame for sampled episodes, or fast forward
                should_render = (ep <= 10) or (ep % 25 == 0) or (ep >= episodes - 5)
                if should_render:
                    frame = visualizer.render_frame(
                        state=state,
                        last_action=action,
                        step_count=step,
                        total_reward=total_reward,
                        status=info,
                        visited_path=visited,
                        q_agent=agent,
                        ep_num=ep
                    )
                    sys.stdout.write("\033[H\033[2J")
                    print(f" {Color.BOLD}WATCHING LIVE Q-LEARNING: Episode {ep}/{episodes} (Epsilon: {agent.epsilon*100:.1f}%){Color.RESET}\n")
                    print(frame)
                    sys.stdout.flush()
                    time.sleep(step_delay)
                    
                state = next_state
                if done:
                    break
                    
            agent.decay_epsilon()
            
    finally:
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()
        
    print(Color.GREEN + f"\n ✔ Live training complete! Agent converged after {episodes} episodes." + Color.RESET)
    time.sleep(1.0)
    return agent


def evolution_showcase(env, visualizer):
    """Shows snapshots of AI performance at different stages of its learning life."""
    stages = [
        (1, 0.95, "Stage 1: Untrained Rookie (Pure Exploration & Random Turning)"),
        (50, 0.60, "Stage 2: Early Learner (Discovering Driving Corridors)"),
        (250, 0.20, "Stage 3: Proficient Driver (Approaching the Parking Stall)"),
        (1500, 0.0, "Stage 4: Master AI (Flawless Autonomous Parking)")
    ]
    
    agent = QLearningAgent(alpha=0.2, gamma=0.98, epsilon=1.0)
    
    print("\n" + Color.CYAN + "═" * 66 + Color.RESET)
    print(f" {Color.BOLD}🚗 AI LEARNING EVOLUTION SHOWCASE{Color.RESET}")
    print(f" Watching how the car's driving behavior evolves across generations...")
    print(Color.CYAN + "═" * 66 + Color.RESET)
    time.sleep(1.2)
    
    current_ep = 0
    for target_ep, eps, description in stages:
        episodes_to_run = target_ep - current_ep
        if episodes_to_run > 0:
            train_agent(env, agent=agent, episodes=episodes_to_run, verbose=False)
            current_ep = target_ep
            
        agent.epsilon = eps
        print(f"\nLaunching: {description} (Press Enter to watch)")
        try:
            input(" > ")
        except (EOFError, KeyboardInterrupt):
            pass
            
        run_animated_episode(env, agent, visualizer, 
                             start_state=(env.start[0], env.start[1], 1),
                             step_delay=0.18, ep_num=target_ep, title=description)
        time.sleep(0.5)


def multi_spot_challenge(env, agent, visualizer, num_tests=5):
    """Test parking autonomy from multiple randomized locations across the lot."""
    print("\n" + Color.CYAN + "═" * 66 + Color.RESET)
    print(f" {Color.BOLD}🎯 RANDOM SPAWN CHALLENGE: 5 ATTEMPTS FROM UNKNOWN SPOTS{Color.RESET}")
    print(Color.CYAN + "═" * 66 + Color.RESET)
    time.sleep(1.0)
    
    successes = 0
    for i in range(1, num_tests + 1):
        sx, sy = random.choice(env.valid_spawns)
        so = random.randint(0, 3)
        title = f"Challenge {i}/{num_tests}: Starting from ({sx}, {sy}) facing {DIR_NAMES[so]}"
        
        ok = run_animated_episode(env, agent, visualizer, 
                                  start_state=(sx, sy, so),
                                  step_delay=0.16, ep_num=i, title=title)
        if ok:
            successes += 1
            print(Color.GREEN + f" ✔ Run {i}: Successfully Parked!" + Color.RESET)
        else:
            print(Color.RED + f" ✘ Run {i}: Failed / Collided." + Color.RESET)
        time.sleep(0.5)
        
    print(Color.BOLD + f"\nResults: {successes}/{num_tests} Successful Autonomous Parks!" + Color.RESET)


# -----------------------------------------------------------------------------
# Main CLI & Interactive Menu
# -----------------------------------------------------------------------------
def print_banner():
    banner = f"""
{Color.CYAN}╔════════════════════════════════════════════════════════════════╗
║    {Color.BOLD}{Color.WHITE}🚗  AUTONOMOUS SELF-PARKING CAR SIMULATOR (PYTHON AI){Color.RESET}{Color.CYAN}       ║
║    {Color.GRAY}Reinforcement Learning (Q-Learning) & Live ASCII Graphics{Color.RESET}{Color.CYAN}   ║
╚════════════════════════════════════════════════════════════════╝{Color.RESET}
"""
    print(banner)

def main():
    parser = argparse.ArgumentParser(description="Autonomous Self-Parking Car with Q-Learning and ASCII Graphics")
    parser.add_argument("--mode", choices=["demo", "live-train", "random", "evolution", "menu"], default="demo",
                        help="Execution mode (default: demo)")
    parser.add_argument("--speed", type=float, default=0.20,
                        help="Animation delay in seconds per step (default: 0.20)")
    parser.add_argument("--no-color", action="store_true",
                        help="Disable ANSI colors")
    args = parser.parse_args()
    
    use_color = not args.no_color
    env = ParkingEnvironment()
    visualizer = AsciiVisualizer(env, use_color=use_color)
    
    if args.mode == "demo":
        print_banner()
        print(" [1/2] Training AI agent on parking lot layout...")
        agent = train_agent(env, episodes=2500, verbose=True)
        print("\n [2/2] Launching autonomous self-parking demo from Entrance [S]...\n")
        time.sleep(0.8)
        run_animated_episode(env, agent, visualizer, step_delay=args.speed, title="Autonomous Parking Demonstration")
        
        # Post-run prompt
        print("\n" + Color.GREEN + Color.BOLD + "🎉 VEHICLE PARKED PERFECTLY IN BAY [P]! 🎉" + Color.RESET)
        print("\nWhat would you like to try next?")
        print("  1. Watch parking from 5 random positions (Multi-Spot Challenge)")
        print("  2. Watch Live Training (See AI learn step-by-step from scratch)")
        print("  3. View Evolution Showcase (Rookie -> Expert comparison)")
        print("  4. Exit")
        try:
            choice = input("\nSelect an option [1-4] (default 1): ").strip()
            if choice == "2":
                live_training_view(env, visualizer, episodes=350, step_delay=0.03)
            elif choice == "3":
                evolution_showcase(env, visualizer)
            elif choice == "4":
                print("Exiting. Goodbye!")
                return
            else:
                multi_spot_challenge(env, agent, visualizer, num_tests=5)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            
    elif args.mode == "live-train":
        print_banner()
        agent = live_training_view(env, visualizer, episodes=350, step_delay=0.03)
        print("Running victory lap with trained AI:")
        run_animated_episode(env, agent, visualizer, step_delay=args.speed, title="Trained Policy Demo")
        
    elif args.mode == "evolution":
        print_banner()
        evolution_showcase(env, visualizer)
        
    elif args.mode == "random":
        print_banner()
        agent = train_agent(env, episodes=2500, verbose=True)
        multi_spot_challenge(env, agent, visualizer, num_tests=5)
        
    elif args.mode == "menu":
        print_banner()
        print("Available Modes:")
        print("  1. Autonomous Parking Demo")
        print("  2. Watch Live Training")
        print("  3. Evolution Showcase")
        print("  4. Random Spawn Challenge")
        choice = input("\nSelect [1-4]: ").strip()
        if choice == "2":
            live_training_view(env, visualizer, episodes=350, step_delay=0.03)
        elif choice == "3":
            evolution_showcase(env, visualizer)
        elif choice == "4":
            agent = train_agent(env, episodes=2500, verbose=True)
            multi_spot_challenge(env, agent, visualizer, num_tests=5)
        else:
            agent = train_agent(env, episodes=2500, verbose=True)
            run_animated_episode(env, agent, visualizer, step_delay=args.speed)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.stdout.write("\033[?25h\n") # Restore cursor
        print("\nSimulation aborted by user.")
