# StreamRL with SPR

Paper : [Squeezing More from the Stream : Learning Representation Online for Streaming Reinforcement Learning](https://www.arxiv.org/abs/2602.09396)

Evaluation of Self-Predictive Representations (SPR) combined with streaming reinforcement learning algorithms on Atari environments.

## Overview

This repository implements and compares baseline streaming RL algorithms with their SPR-augmented variants:

- **DQN** / **DQN-SPR**: Deep Q-Network with optional SPR
- **QRC** / **QRC-SPR**: Quick Replay Candidates with optional SPR  
- **StreamQ** / **StreamQ-SPR**: Streaming Q-learning with optional SPR

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Run an algorithm on an Atari environment:

```bash
python algorithms/dqn.py --env_type atari --env_id ALE/Pong-v5
```
