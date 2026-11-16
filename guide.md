# 🚗 Complete Step-by-Step Guide: Autonomous Self-Parking Car
### Reinforcement Learning (Q-Learning) & Live ASCII Graphics in Python

This guide provides a comprehensive, step-by-step walkthrough of how the autonomous self-parking car simulation in [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py) works from the ground up.

---

## 📑 Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Step 1: The Parking Lot Environment & Grid World](#2-step-1-the-parking-lot-environment--grid-world)
3. [Step 2: Car Kinematics & Motion Constraints](#3-step-2-car-kinematics--motion-constraints)
4. [Step 3: Simulated LIDAR & Proximity Raycasting](#4-step-3-simulated-lidar--proximity-raycasting)
5. [Step 4: The AI Brain — Tabular Q-Learning](#5-step-4-the-ai-brain--tabular-q-learning)
6. [Step 5: Reward Function & Reward Shaping](#6-step-5-reward-function--reward-shaping)
7. [Step 6: Terminal ASCII Graphics Engine](#7-step-6-terminal-ascii-graphics-engine)
8. [Step 7: The Training & Simulation Loops](#8-step-7-the-training--simulation-loops)
9. [Step 8: How to Customize & Experiment](#9-step-8-how-to-customize--experiment)

---

## 1. High-Level Architecture

The project is structured into four decoupled subsystems operating in a classic Reinforcement Learning loop:

```mermaid
flowchart TD
    subgraph Environment ["ParkingEnvironment (Physics & Space)"]
        GRID["2D Grid Map (Walls, Bays, Obstacle Cars)"]
        LIDAR["4-Direction LIDAR Raycaster"]
        PHYSICS["Motion & Collision Detector"]
    end

    subgraph Agent ["QLearningAgent (AI Brain)"]
        QTABLE["Q-Table: Q(state, action)"]
        POLICY["Epsilon-Greedy Action Selector"]
        BELLMAN["Bellman Equation Q-Updater"]
    end

    subgraph Renderer ["AsciiVisualizer (Terminal Display)"]
        ANSI["ANSI Color & Screen Buffer"]
        HUD["Telemetry & Sensor Radar HUD"]
        MAP_DRAW["Top-Down ASCII Map & Tire Tracks"]
    end

    Agent -->|"Action: FWD, REV, LEFT, RIGHT"| PHYSICS
    PHYSICS -->|"Next State, Reward, Done"| Agent
    PHYSICS -->|"State & Collision Info"| Renderer
    LIDAR -->|"Distance Readings"| Renderer
    QTABLE -->|"Action Confidence Q-Values"| Renderer
```

Every frame:
1. The **Agent** observes its current state $(x, y, \theta)$ and consults its **Q-Table**.
2. It executes an action (`FORWARD`, `REVERSE`, `TURN_LEFT`, `TURN_RIGHT`).
3. The **Environment** simulates the movement, checks for collisions, evaluates target bay parking, and calculates the reward $R$.
4. The **LIDAR Raycaster** scans the distance to obstacles in front, rear, left, and right.
5. The **ASCII Visualizer** draws the complete HUD, sensor radar, Q-value confidence meters, and the top-down parking lot map.

---

## 2. Step 1: The Parking Lot Environment & Grid World

In [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L38-L50), the parking lot is modeled as a 2D discrete coordinate grid ($16 \times 8$ cells):

```python
DEFAULT_MAP = [
    "################",  # Row 0: Outer perimeter wall
    "#|C| |C|P|C| | #",  # Row 1: Upper parking stalls (P is target bay at x=8)
    "#              #",  # Row 2: Upper driveway & bay maneuvering corridor
    "#              #",  # Row 3: Central driving aisle
    "#              #",  # Row 4: Central driving aisle
    "#   |C| |C| |C|#",  # Row 5: Island with parked obstacle cars
    "#S             #",  # Row 6: Entrance gate (S at x=1, y=6)
    "################",  # Row 7: Outer perimeter wall
]
```

### Cell Types and Semantics
- `#`: Solid concrete barrier / outer wall (impassable; causes collision).
- `|`: Stall separator curbs (impassable; forces cars to enter stalls from the aisle).
- `C`: Parked obstacle vehicles (impassable; crashing into them ends the run).
- `P`: Designated target parking bay at coordinate $(8, 1)$.
- `S`: Starting entrance gate at coordinate $(1, 6)$.
- `' '`: Driveable asphalt.

### Environment Parsing (`ParkingEnvironment.__init__`)
When initialized, the environment parses the string matrix into fast lookup sets:
- `self.obstacles`: Set of $(x, y)$ coordinates for walls `#`, parked cars `C`, and dividers `|`.
- `self.passable`: Set of $(x, y)$ coordinates where a car can legally drive.
- `self.target`: Exact coordinate of the goal $(8, 1)$.
- `self.start`: Starting entrance $(1, 6)$.

Because the dividers `|` flank the target bay `P` on both the left $(7, 1)$ and right $(9, 1)$, and the back $(8, 0)$ is a solid wall, **it is physically impossible to enter the parking bay sideways**. The car must navigate to $(8, 2)$ in front of the bay, align itself, and pull in!

---

## 3. Step 2: Car Kinematics & Motion Constraints

Real cars have **non-holonomic constraints**—they cannot translate sideways like a chess piece. To simulate authentic vehicular motion, the car maintains both a **position** and a **heading orientation**:

$$\text{State } s = (x, y, \theta)$$

Where $\theta \in \{0, 1, 2, 3\}$ corresponds to:
- `0`: **NORTH** $\uparrow$ (vector $(0, -1)$)
- `1`: **EAST** $\rightarrow$ (vector $(1, 0)$)
- `2`: **SOUTH** $\downarrow$ (vector $(0, 1)$)
- `3`: **WEST** $\leftarrow$ (vector $(-1, 0)$)

### The 4 Actions:
In [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L115-L135), the transition function is implemented in `step()`:

```python
x, y, o = state

if action == 0:    # FORWARD: Advance 1 step in current heading
    nx, ny, no = x + DIRECTIONS[o][0], y + DIRECTIONS[o][1], o
elif action == 1:  # REVERSE: Back up 1 step opposite to current heading
    nx, ny, no = x - DIRECTIONS[o][0], y - DIRECTIONS[o][1], o
elif action == 2:  # TURN_LEFT: Rotate 90° counter-clockwise in place
    nx, ny, no = x, y, (o - 1) % 4
elif action == 3:  # TURN_RIGHT: Rotate 90° clockwise in place
    nx, ny, no = x, y, (o + 1) % 4
```

This means:
- If a car is facing North and executes `FORWARD`, it moves to $(x, y - 1)$.
- If a car wants to move East, it must first execute `TURN_RIGHT` to face East, and then `FORWARD`.
- If a car wants to back into a stall, it can steer past the bay, turn around, and execute `REVERSE`.

---

## 4. Step 3: Simulated LIDAR & Proximity Raycasting

Modern self-parking vehicles (such as Tesla Autopark or Waymo) rely on ultrasonic sensors and LIDAR to measure clearance. 

In [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L90-L113), the method `get_lidar(x, y, heading)` casts 4 rays relative to the vehicle's current orientation:

```mermaid
flowchart TD
    CAR["Autonomous Car (x, y, heading)"]
    CAR -->|"Ray 1: heading"| FRONT["Front Distance"]
    CAR -->|"Ray 2: (heading + 2) % 4"| REAR["Rear Distance"]
    CAR -->|"Ray 3: (heading - 1) % 4"| LEFT["Left Distance"]
    CAR -->|"Ray 4: (heading + 1) % 4"| RIGHT["Right Distance"]
```

### Raycasting Implementation:
```python
def get_lidar(self, x, y, heading):
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
```

### Visual Sensor Gauges:
The distances are formatted into terminal progress bars:
```python
filled = min(distance, 8)
empty  = 8 - filled
bar = "█" * filled + "░" * empty
# Example: [████░░░░] 4m
```
When an obstacle is close ($\le 2\text{m}$), the bar turns red; when clear, it turns green.

---

## 5. Step 4: The AI Brain — Tabular Q-Learning

Rather than using complex neural networks that require heavy frameworks (PyTorch/TensorFlow) and minutes of training, this simulation uses **Tabular Q-Learning**.

### Why Tabular Q-Learning?
- **Guaranteed Convergence**: Mathematically proven to find the optimal path in finite Markov Decision Processes.
- **Ultra-Fast Training**: Trains 3,000 complete episodes in less than **0.05 seconds** in pure Python!
- **Zero Dependencies**: Runs out-of-the-box on standard Python 3.

### The Q-Table
The Q-table is a dictionary mapping each state $s = (x, y, \theta)$ to 4 Q-values, one for each action:
$$Q(s) = [Q_{\text{FWD}}, Q_{\text{REV}}, Q_{\text{LEFT}}, Q_{\text{RIGHT}}]$$

With $16 \times 8 = 128$ grid cells and 4 orientations, there are at most $128 \times 4 = 512$ possible states. A lookup table with 512 entries fits comfortably in memory.

### The Bellman Equation Update
When the agent takes action $a$ in state $s$, receives reward $r$, and transitions to $s'$, it updates its belief:

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big( \underbrace{r + \gamma \max_{a'} Q(s', a')}_{\text{Target}} - \underbrace{Q(s, a)}_{\text{Current Estimate}} \Big)$$

In [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L175-L185):
```python
def update(self, state, action, reward, next_state, done):
    current_q = self.get_q(state)[action]
    if done:
        target = reward
    else:
        best_future = max(self.get_q(next_state))
        target = reward + self.gamma * best_future
        
    self.q_table[state][action] += self.alpha * (target - current_q)
```

Hyperparameters:
- **Learning Rate $\alpha = 0.20$**: Determines how quickly new experience overwrites old estimates.
- **Discount Factor $\gamma = 0.98$**: Balances immediate rewards vs. long-term strategic parking success.

### Epsilon-Greedy Exploration ($\epsilon$-Greedy)
To discover the parking stall initially, the car must explore randomly:
```python
if explore and random.random() < self.epsilon:
    return random.randint(0, 3) # Random exploration
else:
    return argmax(Q[state])     # Exploitation of best known move
```
- Epsilon starts at $\epsilon = 1.0$ (100% random actions).
- After each episode, it decays by $\epsilon \leftarrow \max(\epsilon_{\min}, \epsilon \times 0.998)$.
- By episode 2,000, $\epsilon = 0.02$, meaning the car relies 98% on its learned optimal strategy.

---

## 6. Step 5: Reward Function & Reward Shaping

The reward function guides the agent's behavior:

| Event | Reward | Purpose |
| :--- | :--- | :--- |
| **Parked Head-In ($\theta = \text{North}$)** | **$+280.0$** | Highest reward for correctly pulling into Bay [P]. |
| **Parked Back-In / Any ($\theta \ne \text{North}$)** | **$+220.0$** | Reward for successfully reaching Bay [P]. |
| **Collision (Wall, Divider, or Car)** | **$-60.0$** | Severe penalty; terminates the episode. |
| **Every Step Taken** | **$-1.0$** | Time penalty; forces finding the shortest path. |
| **Distance Shaping** | **$+0.6 \times (D_{\text{old}} - D_{\text{new}})$** | Reward gradient nudging the car toward the stall. |
| **Reversing on Open Road** | **$-0.3$** | Penalizes driving in reverse when cruising forward. |

### Why Distance Shaping is Essential
In pure RL with sparse rewards, the car only gets a positive signal when it randomly stumbles upon the exact parking coordinate $(8, 1)$. By providing a slight gradient bonus whenever the Manhattan distance to the goal decreases:

$$\Delta D = |x_{\text{prev}} - x_{\text{tgt}}| + |y_{\text{prev}} - y_{\text{tgt}}| - \Big( |x_{\text{next}} - x_{\text{tgt}}| + |y_{\text{next}} - y_{\text{tgt}}| \Big)$$

The agent learns 50x faster without getting trapped in endless random loops.

---

## 7. Step 6: Terminal ASCII Graphics Engine

Terminal rendering is handled by [`AsciiVisualizer`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L205-L330).

### 1. Flicker-Free Redrawing
Instead of scrolling text down the screen, the visualizer uses ANSI escape sequences:
- `\033[?25l`: Hides the cursor during playback.
- `\033[H\033[2J`: Moves cursor to row 1, col 1 and clears the screen buffer.
- `\033[?25h`: Restores cursor visibility upon exit or interruption.

### 2. Unicode East-Asian Wide Character Alignment
Emojis like `🚗` occupy **2 terminal columns** on screen, but Python's `len('🚗')` evaluates to **1**. If ignored, standard string `.ljust()` operations will misalign the right box border `│` by 1 space.

To solve this, helper functions inspect Unicode widths:
```python
import unicodedata, re

ANSI_RE = re.compile(r'\033\[[0-9;]*m')

def char_width(c):
    return 2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1

def visible_width(s):
    clean = ANSI_RE.sub('', s)  # Strip invisible ANSI color escape codes
    return sum(char_width(c) for c in clean)

def pad_to(s, target_w):
    w = visible_width(s)
    return s + " " * max(0, target_w - w)
```
This guarantees that all borders (`┌─┐`, `│`, `└─┘`) align across GNOME Terminal, xterm, VS Code terminal, and Apple Terminal.

### 3. Dynamic Tire Tracks
During navigation, the script accumulates visited $(x, y)$ coordinates in a set:
```python
visited = set()
# each step:
visited.add((state[0], state[1]))
```
When rendering the grid:
- If $(x, y) == (x_{\text{car}}, y_{\text{car}})$: render `[^]`, `[>]`, `[v]`, or `[<]`.
- Else if $(x, y) \in \text{visited}$: render yellow tire tracks ` · `.
- Else: render original map element (asphalt, barrier, or wall).

This visually reveals the vehicle's trajectory in real time.

---

## 8. Step 7: The Training & Simulation Loops

### Generalization Training:
In [`train_agent()`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L335-L385):
- 50% of episodes spawn at the entrance gate $(1, 6)$.
- 50% of episodes spawn at **random open cells across the lot**.
- This forces the Q-table to learn a complete navigation policy across the entire parking lot, not just a single hardcoded path.

### Execution Modes:
1. **Demo Mode (`--mode demo`)**:
   Runs background training (2,500 episodes in ~0.04s) and immediately displays an animated self-parking run from the entrance, followed by an interactive menu.
2. **Live Training (`--mode live-train`)**:
   Shows the car learning on-screen frame-by-frame. You watch it stumble into walls in early episodes, discover the driveway in mid episodes, and park smoothly in final episodes.
3. **Random Challenge (`--mode random`)**:
   Tests the trained AI across 5 randomly generated spawn positions to demonstrate 100% autonomous navigation success from anywhere.
4. **Evolution Showcase (`--mode evolution`)**:
   Inspects 4 generational snapshots (Episode 1: Rookie $\to$ Episode 50: Learner $\to$ Episode 250: Proficient $\to$ Episode 1500: Master).

---

## 9. Step 8: How to Customize & Experiment

You can modify [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py) to try new scenarios:

### 1. Creating a New Parking Lot Map
Change `DEFAULT_MAP` in [`autopark.py`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L38-L50):
```python
DEFAULT_MAP = [
    "####################",
    "#|C| |C| |C|P|C| | #",
    "#                  #",
    "#                  #",
    "#   |C|      |C|   #",
    "#S                 #",
    "####################",
]
```
The script will automatically detect dimensions, find `P` and `S`, identify obstacles, and re-train the agent to match the new layout!

### 2. Tuning Animation Speed
Pass `--speed` on the command line:
```bash
# Faster animation (0.08 seconds per move)
python3 autopark.py --speed 0.08

# Slower, cinematic animation (0.35 seconds per move)
python3 autopark.py --speed 0.35
```

### 3. Adjusting Q-Learning Hyperparameters
In [`QLearningAgent`](file:///home/vgalo/Sandbox/machine_learn_car/autopark.py#L150-L160):
- Increase `alpha` (e.g. `0.4`) to make the agent adapt rapidly to changes.
- Increase `gamma` (e.g. `0.99`) for longer-horizon planning.
- Tweak `eps_decay` to control how fast the car transitions from random exploration to exploitation.
