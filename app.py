import gymnasium as gym
import time
from ale_py import ALEInterface
#from ale_py.roms import Breakout
from agent import RandomAgent
import matplotlib.pyplot as plt

def main():
    # Create the Atari environment
    env = gym.make("ALE/Breakout-v5", render_mode='rgb_array')
    
    # Create a random agent
    agent = RandomAgent(env)
    
    # Run a random episode
    print("Running a random episode...")
    total_reward, frames = agent.run_episode(max_steps=5000, render=True)
    print(f"Episode complete. Total reward: {total_reward}")
    
    # Display some frames as a sanity check
    if frames:
        plt.figure(figsize=(10, 5))
        for i in range(min(5, len(frames))):
            plt.subplot(1, 5, i+1)
            plt.imshow(frames[i], cmap='gray')
            plt.title(f"Frame {i}")
            plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    # Close the environment
    env.close()

if __name__ == "__main__":
    main()