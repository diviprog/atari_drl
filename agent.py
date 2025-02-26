import cv2

class RandomAgent:
    def __init__(self, env):
        """Initialize the random agent with an Atari environment.
        
        Args:
            env: The Atari environment
        """
        self.env = env
        self.action_space = env.action_space
        
    def select_action(self):
        """Randomly select an action from the environment's action space.
        
        Returns:
            int: A random action
        """
        return self.action_space.sample()
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame (as described in the DQN paper).
        
        Args:
            frame: Raw frame from the environment
            
        Returns:
            Preprocessed frame (84x84 grayscale)
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Resize to 84x84
        resized_frame = cv2.resize(gray_frame, (84, 84), interpolation=cv2.INTER_AREA)
        
        # Normalize pixel values
        normalized_frame = resized_frame / 255.0
        
        return normalized_frame
        
    def run_episode(self, max_steps=1000, render=True):
        """Run one episode with the random agent.
        
        Args:
            max_steps: Maximum number of steps to run
            render: Whether to render the environment
            
        Returns:
            total_reward: The total reward for the episode
            frames: List of preprocessed frames if render is True
        """
        observation, info = self.env.reset()
        total_reward = 0
        frames = []
        
        for step in range(max_steps):
            if render:
                # Store the preprocessed frame
                preprocessed_frame = self.preprocess_frame(observation)
                frames.append(preprocessed_frame)
            
            # Select random action
            action = self.select_action()
            
            # Take action in environment
            observation, reward, terminated, truncated, info = self.env.step(action)
            
            # Accumulate reward
            total_reward += reward
            
            # Check if episode is done
            if terminated or truncated:
                break
                
        return total_reward, frames