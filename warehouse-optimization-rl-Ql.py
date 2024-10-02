import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

# Custom Gym Environment for Warehouse Optimization
class WarehouseEnv(gym.Env):
    def __init__(self):
        super(WarehouseEnv, self).__init__()
        
        # Define action and observation space
        self.action_space = gym.spaces.Discrete(3)  # 0: Do nothing, 1: Restock, 2: Sell
        self.observation_space = gym.spaces.MultiDiscrete([101, 10, 5])  # Inventory (0-100), Demand (0-9), Price (5 levels)
        
        # Initialize state
        self.reset()
        
    def reset(self):
        self.inventory = 50
        self.demand = np.random.randint(0, 10)
        self.price = np.random.randint(0, 5)  # 5 discrete price levels
        return self._get_state()
    
    def step(self, action):
        # Apply action
        if action == 1:  # Restock
            self.inventory = min(100, self.inventory + 10)
        elif action == 2:  # Sell
            self.inventory = max(0, self.inventory - self.demand)
        
        # Calculate reward
        if action == 2:
            reward = self.demand * (self.price * 0.2 + 0.8)  # Price levels 0-4 map to 0.8-1.6
        else:
            reward = 0
        
        # Update state
        self.demand = np.random.randint(0, 10)
        self.price = np.random.randint(0, 5)
        
        # Check if done
        done = False
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        return np.array([self.inventory, self.demand, self.price])

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.1
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_table = torch.zeros((*state_size, action_size), device=self.device)
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.long)
        return torch.argmax(self.q_table[state_tensor[0], state_tensor[1], state_tensor[2]]).item()
    
    def learn(self, state, action, reward, next_state):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.long)
        next_state_tensor = torch.tensor(next_state, device=self.device, dtype=torch.long)
        
        current_q = self.q_table[state_tensor[0], state_tensor[1], state_tensor[2], action]
        max_next_q = torch.max(self.q_table[next_state_tensor[0], next_state_tensor[1], next_state_tensor[2]])
        new_q = current_q + self.learning_rate * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_tensor[0], state_tensor[1], state_tensor[2], action] = new_q
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_optimal_action(self, state):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.long)
        return torch.argmax(self.q_table[state_tensor[0], state_tensor[1], state_tensor[2]]).item()

    def get_optimal_inventory(self, demand, price):
        optimal_inventories = []
        for inventory in range(101):  # 0 to 100
            state = [inventory, demand, price]
            action = self.get_optimal_action(state)
            if action == 1:  # Restock
                optimal_inventories.append(inventory)
        return optimal_inventories if optimal_inventories else [100]  # If empty, suggest max inventory


# Training the model
env = WarehouseEnv()
state_size = env.observation_space.nvec
action_size = env.action_space.n
agent = QLearningAgent(state_size, action_size)

episodes = 10000
rewards_history = []

for e in range(episodes):
    state = env.reset()
    total_reward = 0
    
    for time in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state)
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    rewards_history.append(total_reward)
    if (e + 1) % 100 == 0:
        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}, Epsilon: {agent.epsilon:.2f}")

# Visualization of training progress
plt.figure(figsize=(10, 5))
plt.plot(rewards_history)
plt.title('Rewards per Episode')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()

# Test the trained model
test_episodes = 10
for e in range(test_episodes):
    state = env.reset()
    total_reward = 0
    inventory_history = []
    demand_history = []
    price_history = []
    
    for time in range(100):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        
        inventory_history.append(state[0])
        demand_history.append(state[1])
        price_history.append(state[2])
        
        if done:
            break
    
    print(f"Test Episode: {e+1}/{test_episodes}, Total Reward: {total_reward}")
    
    # Visualization for each test episode
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(inventory_history)
    plt.title('Inventory Level')
    plt.xlabel('Time Step')
    plt.ylabel('Inventory')
    
    plt.subplot(1, 3, 2)
    plt.plot(demand_history)
    plt.title('Demand')
    plt.xlabel('Time Step')
    plt.ylabel('Demand')
    
    plt.subplot(1, 3, 3)
    plt.plot(price_history)
    plt.title('Price Level')
    plt.xlabel('Time Step')
    plt.ylabel('Price')
    
    plt.tight_layout()
    plt.show()
    
# Function to get inventory recommendation
def get_inventory_recommendation(agent, demand, price):
    optimal_inventories = agent.get_optimal_inventory(demand, price)
    min_optimal = min(optimal_inventories)
    max_optimal = max(optimal_inventories)
    
    if min_optimal == max_optimal:
        return f"For demand {demand} and price level {price}, the recommended inventory level is {min_optimal}."
    else:
        return f"For demand {demand} and price level {price}, the recommended inventory range is {min_optimal}-{max_optimal}."

# Example usage
print(get_inventory_recommendation(agent, demand=5, price=2))
print(get_inventory_recommendation(agent, demand=8, price=4))

# Visualize optimal inventory levels for different demand and price combinations
demand_range = range(10)
price_range = range(5)

plt.figure(figsize=(12, 8))
for price in price_range:
    optimal_inventories = [min(agent.get_optimal_inventory(demand, price)) for demand in demand_range]
    plt.plot(demand_range, optimal_inventories, label=f'Price Level {price}')

plt.title('Optimal Inventory Levels')
plt.xlabel('Demand')
plt.ylabel('Optimal Inventory')
plt.legend()
plt.grid(True)
plt.show()
