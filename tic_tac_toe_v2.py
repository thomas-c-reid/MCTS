import numpy as np
import gym
from time import sleep
from uuid import uuid4
from copy import deepcopy
from datetime import datetime
from dataclasses import dataclass
from gym.envs.registration import register

# Will need a better find_node function to check if state has been visited but with opposite player values
# it will also adding new nodes on each iteration - I think this means we might be reseting the list of nodes at some point
# -- long story short we need to look into the find_node function

register(
    id='tictactoe-v0',
    entry_point='tictactoe_gym.envs.tictactoe_env:TicTacToeEnv',
)

@dataclass
class Action:
    """
    small data class to hold information on an action to be taken in the environment
    
    ATTRIBUTES:
    action (int): The value of the action taken in the environment (1-9), corresponding to board positions.
    player_turn (int): Indicates which player took the action (-1 for one player, 1 for the other).
    tree_node_id (uuid): The unique identifier of the TreeNode associated with this action.
    """
    action: int
    player_turn: int
    tree_node_id: uuid4
    value: int

class GameState:
    """
    This will be a representation of the current game of tictactoe 
    
    ATTRIBUTES:
        - state: (NARRAY) the game space representation of the current observation
        - player_turn: (INT) the current turn of the player in the round (-1 or 1) 
        - is_terminal: (BOOL) True if game is in terminal state otherwise False
        - winner: (INT) If game is in terminal state, winner will be player who gets reward value (-1, 0 or 1)
    """
    def __init__(self, observation_space: tuple):
                
        if len(observation_space) == 2:
            self.state = observation_space[0]
            self.player_turn = observation_space[1].get('player')
            self.is_terminal, self.winner = self.is_terminal(self.state)
        elif len(observation_space) == 5:
            self.state = observation_space[0]
            self.player_turn = observation_space[-1].get('player')
            self.is_terminal, self.winner = self.is_terminal(self.state)
                
    @staticmethod
    def is_terminal(state_representation):
        """
        Helper function that takes a 3x3 state representation and checks if the game state is terminal and who the winner is.
        The winner can be -1 (player -1 wins), 1 (player 1 wins), or 0 (draw).
        
        Args:
        state_representation (np.array): 3x3 numpy array representing the Tic-Tac-Toe board.

        Returns:
        tuple: (is_terminal, winner) where is_terminal is a boolean and winner is -1, 0, or 1.
        """
        
        winner = None
        is_terminal = False
        
        # Check rows and columns for a winner
        for i in range(3):
            # Check rows
            if np.all(state_representation[i, :] == 1):
                winner = 1
                is_terminal = True
            elif np.all(state_representation[i, :] == -1):
                winner = -1
                is_terminal = True
            
            # Check columns
            if np.all(state_representation[:, i] == 1):
                winner = 1
                is_terminal = True
            elif np.all(state_representation[:, i] == -1):
                winner = -1
                is_terminal = True

        # Check diagonals for a winner
        if np.all(np.diag(state_representation) == 1) or np.all(np.diag(np.fliplr(state_representation)) == 1):
            winner = 1
            is_terminal = True
        elif np.all(np.diag(state_representation) == -1) or np.all(np.diag(np.fliplr(state_representation)) == -1):
            winner = -1
            is_terminal = True

        # Check for a draw (no winner and no empty spaces)
        if winner is None and not np.any(state_representation == 0):
            winner = 0
            is_terminal = True
        
        # print('ASDFASDFASDFASDF')
        # print(is_terminal, winner)
        return is_terminal, winner
    
    def __str__(self):
        string = f"\n {self.state} /n {self.is_terminal}\n"
        return string
    
class TreeNode:
    """
    This class represents a single node in the tree. Each node is a possible game state in the Monte Carlo Tree Search (MCTS) tree,
    holding all necessary information for the MCTS algorithm to operate effectively.

    ATTRIBUTES:
    id: (uuid4) A unique identifier for each node in the tree.
    gameState: (GameState) The game state representation associated with this node.
    n_visit: (int) The number of times this node has been visited during the tree traversal.
    total_reward: (float) The cumulative reward obtained from all simulations passing through this node.
    value: (float) The value calculated for this node using the UCT (Upper Confidence Bound applied to Trees) formula.
    parent: (TreeNode or None) The parent node representing the previous game state; None if this node is the root.
    is_expanded: (bool) Whether the node has been expanded, i.e., its child nodes have been generated.
    children: (list of TreeNode) A list containing the child nodes representing possible future game states resulting from actions taken from this node.
    action_mapping: (list of dicts) dict will contains values to map possible actions to potentials future game states
    """
    
    def __init__(self, observation_space, parent=None):
        self.id = uuid4()
        self.GameState = deepcopy(GameState(observation_space))
        self.n_visits = 0
        self.total_reward = 0
        self.value = 0
        self.parent = parent
        self.is_expanded = False
        self.children = []
        self.action_mapping = []
        
    def update_value(self, reward):
        """
        New value derived from average reward per visit
        """
        self.n_visits += 1
        self.total_reward += reward
        self.value = reward / self.n_visits
        
    def get_action_mask(self):

        state = self.GameState.state
        
        available_spaces = []
        
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    index = i * 3 + j
                    available_spaces.append(index)
        
        return available_spaces
        
class TreeGraph:
    """
    A class to keep track of TreeNode instances and retrieve them based on their state or unique ID.
    
    ATTRIBUTES:
    self.nodesL (list of TreeNodes) A list to stroe all nodes in the tree graph
    
    METHODS:
    add_node(node):
        Adds a TreeNode to the tree graph.

    find_node(state=None, id=None):
        Finds and returns a TreeNode matching the given state or ID.
    """
    
    def __init__(self):
        self.nodes = []
        
    def add_node(self, node: TreeNode):
        self.nodes.append(node)
        print(f'Node added: {len(self.nodes)}')
        
    def find_node(self, state: np.array = None, node_id: uuid4 = None):
        """
        gives the option to search the tree for a TreeNode based off id and GameState
        """
        
        node = None
        found_node = False
        
        while not found_node:
            for tree_node in self.nodes:
                
                if state:
                    if np.array_equal(tree_node.GameState.state, state.state):
                        node = tree_node
                        found_node = True   
                         
                elif node_id:
                    if tree_node.id == node_id:
                        node = tree_node
                        found_node = True
            break
        
        if state and not node:
            node = TreeNode(state)
            self.add_node(node)
        
        return node
        
class MCTS:
    
    def __init__(self, env_name: str, exploration_term: int):
        self.env = gym.make(env_name)
        self.tree_graph = TreeGraph()
        self.root_node = TreeNode(self.env.reset())
        self.tree_graph.add_node(self.root_node)
        self.C = exploration_term
        self.actions_taken = []

    def selection(self, node: TreeNode,  player_turn: int, find_best: bool = True):
        
        action_mappings = []
        
        if not find_best:
            
            parent_n_visit = node.n_visits + 1
            
            for item in node.children:
                
                child_node = item['node']
                action = item['action']
                
                utc_value = 0
                
                if child_node.n_visits == 0:
                    action_mapping = Action(action, player_turn, child_node.id, utc_value)
                    action_mappings.append(action_mapping)
                    continue
                
                child_reward = child_node.total_reward
                
                exploitation_value = child_reward/(child_node.n_visits + 1)
                
                exploration_value = self.C * np.sqrt(np.log(parent_n_visit) / (child_node.n_visits + 1))
                
                utc_value = exploitation_value + exploration_value
                
                action_mapping = Action(action, player_turn, child_node.id, utc_value)
                action_mappings.append(action_mapping)
        else:
            for item in node.children:
                child_node = item['node']
                action = item['action']
                # child_mapping = {'node_id': child_node.id, 'value': child_node.value, 'action': action}
                action_mapping = Action(action, player_turn, child_node.id, child_node.value)
                action_mappings.append(action_mapping)
                        
        # Filter for only available moves
        if len(action_mappings) > 0:
            action_mappings = [action_mapping for action_mapping in action_mappings if action_mapping.action in node.get_action_mask()]
        else:
            print('yo')
        # Find maximum value
        best_child = max(action_mappings, key=lambda x: x.value)
        
        return best_child
        
    def expansion(self, node: TreeNode):
        
        for action in node.get_action_mask():            
            env_ = deepcopy(self.env)
            tree_node = TreeNode(env_.step(action), node)
            node.children.append({'node': tree_node, 'action': action})
            self.tree_graph.add_node(tree_node)
            node.is_expanded = True
    
    def backpropagation(self, state: GameState, actions: list, update_values: bool = True):        
        winner = state.winner
        
        for action in actions:
            node: TreeNode = self.tree_graph.find_node(node_id=action.tree_node_id)
            if action.player_turn == winner:
                reward = 1
            elif action.player_turn == -winner:
                reward = -1
            elif winner == 0:
                reward = 0
            
            node.update_value(reward)
        
        self.env.reset()
                        
    def think(self, state: GameState, time_allowed: int = 30):
        
        start_time = datetime.now()
                
        original_state = state
        
        while (datetime.now() - start_time).total_seconds() < time_allowed:  # Convert timedelta to seconds
            
            actions = []
            state = original_state
            is_terminated = False
            while not is_terminated:
                
                # Get the TreeNode that this game represents
                node = self.tree_graph.find_node(state)
                
                if not node.is_expanded:
                    self.expansion(node)
                    
                best_action = self.selection(node, state.player_turn, find_best=False)
                actions.append(best_action)
                                
                action = best_action.action
                
                # print('action', action)
                    
                observation = self.env.step(action)
                state = GameState(observation)
                # state = GameState(self.env.step(action))
                
                # print('-'*50)
                # print(observation)
                # print(state.state)
                # print(state.is_terminal)
                # print('-'*50)
                                                        
                if state.is_terminal:
                    is_terminated = True
                    print('*'*60)
                    print('Found Terminal State')
                    print(state.state)
                    print('*'*60)
            
            self.backpropagation(state, actions, update_values=True)
                
        # Pick the best option
        node = self.tree_graph.find_node(original_state)
        
        if not node.is_expanded:
            self.expansion(node)
            
        best_action = self.selection(node, state.player_turn, find_best=True)
        return best_action
    
class TrainingEnv:
    
    def __init__(self, env_name: str, training_iterations: int, exploration_term: int, thinking_time: int):
        self.env = gym.make(env_name)
        self.training_iterations = training_iterations
        self.root_node = TreeNode(self.env.reset())
        self.MCTS = MCTS(env_name, exploration_term)
        self.thinking_time = thinking_time
        
    def self_play(self):
        
        for _ in range(self.training_iterations):
            state = GameState(self.env.reset())
            is_terminated = False
            
            while not is_terminated:
                
                action = self.MCTS.think(state, time_allowed=self.thinking_time)
                
                print('ACTION TAKEN: ', action.action)
                                
                state = GameState(self.env.step(action.action))
                
                print(f'NEW STATE: {state.state}', state.is_terminal)
            
            self.MCTS.backpropagation(state, update_values=False)
                
config = {
    'env_name': 'tictactoe-v0',
    'training_iterations': 1000,
    'exploration_term': np.sqrt(2),
    'thinking_time': 1
}

TrainingEnv = TrainingEnv(**config)
TrainingEnv.self_play()
    