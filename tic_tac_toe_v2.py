import numpy as np
import gym
from copy import deepcopy
from uuid import uuid4
from dataclasses import dataclass
from gym.envs.registration import register


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
            self.is_terminal = True if observation_space[1].get('winner') != 0 else False
            if self.is_terminal:
                self.winner = observation_space[1].get('winner')
            else:
                self.winner = None
        elif len(observation_space) == 5:
            self.state = observation_space[0]
            self.player_turn = observation_space[-1].get('player')
            self.is_terminal = True if observation_space[-1].get('winner') != 0 else False
            if self.is_terminal:
                self.winner = observation_space[-1].get('winner')
            else:
                self.winner = None     
                
    def __str__(self):
        string = f"{'<>'*50} \n {self.state} \n {'<>'*50}"
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
        self.GameState = GameState(observation_space)
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
        print(f'node added to tree: size {len(self.nodes)}')
        
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
                    if tree_node.id == tree_node:
                        node = tree_node
                        found_node = True
            break
        return node
        
class MCTS:
    
    def __init__(self, env_name: str, exploration_term: int):
        self.env = gym.make(env_name)
        self.tree_graph = TreeGraph()
        self.root_node = TreeNode(self.env.reset())
        self.tree_graph.add_node(self.root_node)
        self.C = exploration_term
        self.actions_taken = []

    def selection(self, node: TreeNode, find_best: bool = True):
        
        child_mappings = []
        
        if not find_best:
            best_action = None
            
            parent_n_visit = node.n_visits
            
            for child_node in node.children:
                
                utc_value = 0
                
                if child_node.n_visits == 0:
                    child_mapping = {
                        'node_id': child_node.id,
                        'value': utc_value
                    }
                
                child_reward = child_node.total_reward
                
                exploitation_value = child_reward/child_node.n_visits
                
                exploration_value = self.C * np.sqrt(np.log(parent_n_visit) / child_node.n_visits)
                
                child_mapping = {
                    'node_id': child_node.id,
                    'value': utc_value
                }
                child_mappings.append(child_mapping)
        else:
            for child_node in node.children:
                child_mapping = {'node_id': child_node.id, 'value': child_node.value}
                child_mappings.append(child_mapping)
                
        print(child_mappings)
        best_child = max(child_mappings, key=lambda x: x['value'])
        
        return best_child['node_id']
        
    
    def expansion(self, node: TreeNode):
                        
        for action in range(self.env.action_space.n):
            
            env_ = deepcopy(self.env)
            tree_node = TreeNode(env_.step(action), node)
            node.children.append(tree_node)
            self.tree_graph.add_node(tree_node)
                    
    def simulation(self):
        pass
    
    def backpropagation(self):
        pass
    
    def think(self, state: GameState):
        # From the current GameState take a look at the possible actions that could be taken
        # run the selection process until it enters either an unexplored treenode or a terminal state
        # will output an action dataclass
        
        is_terminated = False
        
        
        while not is_terminated:
            
            # Get the TreeNode that this game represents
            node = self.tree_graph.find_node(state)
            
            if not node.is_expanded: 
                self.expansion(node)
                
            best_action = self.selection(node)
            
        
        pass
    
class TrainingEnv:
    
    def __init__(self, env_name: str, training_iterations: int, exploration_term: int):
        self.env = gym.make(env_name)
        self.training_iterations = training_iterations
        self.root_node = TreeNode(self.env.reset())
        self.MCTS = MCTS(env_name, exploration_term)
        
    def self_play(self):
        
        for _ in range(self.training_iterations):
            state = GameState(self.env.reset())
            is_terminated = False
            
            while not is_terminated:
                
                action = self.MCTS.think(state)
                
                state = self.env.step(action.action)
                
        
config = {
    'env_name': 'tictactoe-v0',
    'training_iterations': 1000,
    'exploration_term': np.sqrt(2)
}

TrainingEnv = TrainingEnv(**config)
TrainingEnv.self_play()
    