# Lane Dodge: a DQN built from scratch

A tiny one-lane driving game and a reinforcement learning agent that teaches itself to play it.
The neural network, backpropagation, Adam optimizer and the whole DQN training loop are written by
hand in NumPy. No PyTorch, no RL libraries.

A random agent survives about 13 steps. After about 6,000 training steps (roughly 400 games) the
agent survives the full 1,000-step limit on every test road.

## The game

- A 5-lane road, 9 rows tall. Blocks fall one row per step.
- The car sits on the bottom row and has 3 actions: **LEFT**, **STAY**, **RIGHT**.
- **Front crash**: a block lands on the car. **Side crash**: the car drives into a lane where a
  block is still beside it.
- At most one block per row, never in a column next to the previous row's block, so a safe move
  always exists.

What the agent sees (the state) is 50 numbers: the 9 × 5 grid (1 = block, 0 = empty) plus a
one-hot vector for the car's lane. Rewards: **+1** for every step survived, **−10** for a crash.

## Setup

Python 3.10 or newer.

```
pip install -r requirements.txt
```

`torch` in requirements.txt is optional (only for a later PyTorch comparison); the agent does not
use it. `imageio-ffmpeg` is only needed to export MP4 videos.

## Run it

All commands run from this folder.

| Step | Command | What it does |
|---|---|---|
| Play it yourself | `python play.py` | ← → or A/D to move, SPACE to start, P for a screenshot, ESC to quit |
| Check the maths | `python test_nn.py` | Gradient checks, an overfit test and a short training smoke test |
| Train | `python train.py` | About 1–2 minutes on a CPU, no window. Saves to `runs/dqn/` |
| Watch it learn | `python watch.py` | Montage: untrained → 3 learning stages → final brain (3 runs) |
| Watch the final agent | `python watch.py --ckpt runs/dqn/final.npz --episodes 5 --max-watch-steps 1000` | Just the trained agent, full-length games |
| Record a video | `python watch.py --video lane_dodge.mp4` | Same montage, saved as an H.264 MP4 |

`watch.py` only replays saved brains, so run `train.py` once first (training output is not in git).

Keys in `watch.py`: SPACE skips, ↑/↓ change speed, P saves a screenshot, ESC quits.

### Useful options

```
python play.py --cols 7 --no-speedup        # wider road, constant speed
python play.py --ramp 0.8                   # speed up more slowly (default 1.0 ms per step)
python train.py --steps 150000              # train longer
python train.py --name gamma09 --gamma 0.9  # separate run folder with a different gamma
python watch.py --run gamma09               # watch that run
python record_clips.py --out clips          # exact-length MP4 clips of each stage (for videos)
```

## Files

| File | Job |
|---|---|
| `lane_env.py` | Game rules only (no graphics): `reset()`, `step(action)`, the state vector |
| `renderer.py` | Pygame drawing: road, car, blocks, HUD, optional stats panel |
| `play.py` | Play the game with the keyboard |
| `nn_numpy.py` | The neural network from scratch: forward pass, backpropagation, Adam |
| `dqn_agent.py` | DQN: ε-greedy actions, replay buffer, target network, Huber loss |
| `train.py` | Training loop, checkpoints, test scores, learning-curve plot |
| `watch.py` | Replay saved brains in the game window, montage, MP4 export |
| `test_nn.py` | Proves the hand-written gradients are correct |
| `record_clips.py` | Renders exact-length clips offline (used for the explainer video) |

## How it learns (DQN in one screen)

```
network:  50 inputs → 64 → 64 → 3 outputs   (Q-values for LEFT, STAY, RIGHT; 7,619 weights)

act:      random action with probability ε (1.0 → 0.05 over 30,000 steps), else argmax Q(s, a)
target:   y = r + γ · (1 − crashed) · max Q_target(s′, a′)        γ = 0.95
loss:     Huber(Q(s, a) − y), averaged over a batch of 64 from the replay buffer (50,000 steps)
update:   backpropagation + Adam (lr 0.001), gradient norm clipped to 10
target network: copied from the online network every 1,000 steps
```

Only a crash counts as "done" in the target. Hitting the 1,000-step time limit does not, because
the car was still safe.

## Training output (`runs/<name>/`)

| Path | Contents |
|---|---|
| `checkpoints/ckpt_XXXXXXX.npz` | Brain snapshots: every 500 steps up to 10k, then every 2,500 |
| `final.npz`, `best.npz` | Last brain and best-tested brain |
| `history.json` | Steps survived per episode, checkpoint test scores, settings |
| `learning_curve.png` | Training survival and test score plots |

Training survival levels off around 100–200 steps while the test score is a perfect 1,000. That is
expected: during training the agent still takes a random action 5% of the time, and one random
step into a block ends the game. Tests run with no randomness.
