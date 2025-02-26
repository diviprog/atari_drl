# Atari Deep Reinforcement Learning

This project implements Deep Reinforcement Learning techniques to play Atari 2600 games, based on the groundbreaking paper ["Playing Atari with Deep Reinforcement Learning"](https://arxiv.org/abs/1312.5602) by Mnih et al.

## Overview

We're building a CNN-based Q-network that learns to play Atari games directly from raw pixel inputs. The agent will observe the game screen, process the pixels, and learn optimal policies through reinforcement learning techniques.

## Project Structure

- `random_agent.py`: Implementation of a random baseline agent
- (More modules will be added as the project progresses)

## Features

- 📊 Frame preprocessing (grayscale, resize to 84×84)
- 🔄 Frame stacking (4 frames for temporal context)
- 🧠 CNN model architecture
- 💾 Experience replay buffer
- 📈 Q-learning with epsilon-greedy exploration

## Requirements

- Python 3.7+
- PyTorch or TensorFlow
- OpenAI Gym
- Arcade Learning Environment (ALE)
- OpenCV
- NumPy
- Matplotlib

## Installation

```bash
# Clone the repository
git clone https://github.com/diviprog/atari_drl.git
cd atari_drl

# Set up a virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install gym ale-py opencv-python matplotlib numpy torch
# Install ROMs (you need to manually download Atari ROMs and install them)
```

## Usage

### Running the Random Agent

```python
python random_agent.py
```

This will execute a random agent playing Breakout, serving as a baseline for comparison.

## Implementation Plan

### Phase 1: Environment and Data Pipeline ✅
- Set up ALE
- Implement frame preprocessing
- Create random baseline agent

### Phase 2: Model Development 🔄
- Build CNN model architecture
- Implement experience replay
- Set up Q-learning algorithm

### Phase 3: Training
- Train the model for 10M frames
- Tune hyperparameters
- Save checkpoints

### Phase 4: Evaluation
- Compare against random agent
- Benchmark against baseline methods
- Compare with human scores

### Phase 5: Optimization & Extensions
- Implement Double Q-learning
- Explore Dueling DQN
- Add Prioritized Experience Replay

## License

MIT

## Acknowledgements

- [DeepMind](https://deepmind.com/) for the original DQN research
- [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment)
- [OpenAI Gym](https://gym.openai.com/)