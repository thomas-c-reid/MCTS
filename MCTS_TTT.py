import gym
import logging
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from uuid import uuid4
from copy import deepcopy
from random import choice
from datetime import datetime
import matplotlib.pyplot as plt
from dataclasses import dataclass
from gym.envs.registration import register

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
    utc_value (int): The UTC value for the child node
    """
    action: int
    player_turn: int
    tree_node_id: uuid4
    utc_value: int
    

class Logger:
    """
    A small reusable logging class that provides separate logging configurations for different parts of the application.

    This class allows for easier viewing of logs from different areas of the training process by creating loggers that
    write to separate log files. It ensures that the log file is cleared each time the logger is initialized, so that
    logs from previous runs do not accumulate.

    Attributes:
        logger (logging.Logger): The configured logger instance.

    Methods:
        get_logger():
            Returns the configured logger instance.
    """

    def __init__(self, logger_name: str, log_file: str, level=logging.INFO, console_output=False):
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(level)

        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(level)

        self.logger.addHandler(file_handler)

        if console_output:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
    

class GameState:
    """
    This class represents the current state of a Tic-Tac-Toe game.

    ATTRIBUTES:
        - state (np.array): The game board representation of the current observation.
        - player_turn (int): The current player's turn in the round (-1 or 1).
        - is_terminal (bool): True if the game is in a terminal state; otherwise False.
        - winner (int): If the game is in a terminal state, the winner will be -1, 0, or 1.

    METHODS:
        - __init__(): Initializes the GameState with the given observation space.
        - is_terminal(): Static method that checks if the game state is terminal and determines the winner.
        - __str__(): Returns a string representation of the game board.
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
            
        return is_terminal, winner
    
    def __str__(self):
        # Create a string representation of the board with 'x', 'o', and ' ' for 1, -1, and 0 respectively
        rows = [" | ".join(['x' if cell == 1 else 'o' if cell == -1 else ' ' for cell in row]) for row in self.state]
        grid = "\n" + "\n--+---+--\n".join(rows) + "\n"
        return f"{grid}"
    
    
class TreeNode:
    """
    This class represents a single node in the Monte Carlo Tree Search (MCTS) tree.

    ATTRIBUTES:
        - id (uuid4): A unique identifier for each node in the tree.
        - GameState (GameState): The game state associated with this node.
        - n_visits (int): The number of times this node has been visited during the tree traversal.
        - total_reward (float): The cumulative reward obtained from all simulations passing through this node.
        - value (float): The value calculated for this node.
        - parent (TreeNode or None): The parent node; None if this node is the root.
        - is_expanded (bool): Indicates whether the node has been expanded (child nodes generated).
        - children (list of TreeNode): A list of child nodes representing possible future game states.
        - action_mapping (list of dict): Contains mappings of possible actions to potential future game states.

    METHODS:
        - __init__(): Initializes the TreeNode with the given observation space and parent node.
        - update_value(): Updates the node's value based on the reward received.
        - get_action_mask(): Returns the list of available actions from the current game state.
        - get_utc_value(): Calculates and returns the UCT (Upper Confidence Bound applied to Trees) value for the node.
        - __str__(): Returns a string representation of the node, including the game board and UCT values of child nodes.
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
    
    def get_utc_value(self, parent_visits: int = 0, exploration_term: int = np.sqrt(2), verbose: bool = False):
        
        exploitation_value = self.total_reward / self.n_visits if self.n_visits > 0 else 0.01
        
        exploration_value = exploration_term * np.sqrt(np.log(parent_visits + 1) / (self.n_visits + 0.001))
        
        utc_value = exploitation_value + exploration_value
        
        if verbose:
            print('exploration_value: ', exploration_value)
            print('exploitation_value: ', exploitation_value)
        return utc_value

    def __str__(self):
        # Create a 3x3 grid initialized with "N/A"
        board_representation = [["N/A" for _ in range(3)] for _ in range(3)]
        
        # Populate the grid with utc_values and visit counts from the children
        for child_dict in self.children:
            child = child_dict['node']
            action = child_dict['action']
            
            if isinstance(child, TreeNode) and action is not None:
                i, j = divmod(action, 3)  # Convert the action index to row and column
                utc_value = child.get_utc_value(self.n_visits)
                board_representation[i][j] = f"{utc_value:.2f} ({child.n_visits})"
        
        # Create a string representation of the board with details
        rows = [" | ".join([cell for cell in row]) for row in board_representation]
        grid = "\n" + "\n---+---+---\n".join(rows) + "\n"
        return f" - - - \n {grid} \n {self.GameState} \n {self.id} \n {self.n_visits} - - - "
     
        
class TreeGraph:
    """
    This class represents the game tree used in the Monte Carlo Tree Search (MCTS) algorithm for Tic-Tac-Toe.

    ATTRIBUTES:
        - nodes (list of TreeNode): A list containing all the nodes in the tree.
        - root_node (TreeNode or None): The root node of the tree.

    METHODS:
        - __init__(): Initializes the TreeGraph with an empty node list and no root node.
        - add_node(): Adds a new node to the tree.
        - find_node(): Searches for a node in the tree based on game state, actions, or node ID.
        - display_estimated_values(): Displays the estimated UCT values of child nodes for visualization.
    """
    def __init__(self):
        self.nodes = []
        self.root_node = None
        
    def add_node(self, node: TreeNode):
        if len(self.nodes) == 0:
            self.root_node = node
        self.nodes.append(node)
        
    def find_node(self, state=None, actions=None, node_id=None, verbose=False):
        """
        This function should take a game state and a list of previous actions, and use these actions to 'search'
        the game tree for the given tree node
        
        IF state supplied, actions will also be supplied - else node_id must be supplied
        """
        
        if verbose:
            print('-'*50)
            print('STARTING SEARCH')
            print('State:', state)
            print('type: ', type(state))
            if actions is not None:
                print('actions: ', len(actions))
                print([action.action for action in actions])
            print('root node: ', self.nodes[0].GameState.state)
            print('node_id:', node_id)
            print('number of nodes: ', len(self.nodes))
            print('-'*50)
        
        node = None
        found_node = False
        
        # Search by state
        if state is not None:
            
            # Check if state == root node
            if np.array_equal(self.nodes[0].GameState.state, state):
                return self.nodes[0]
            else:
                current_node = self.nodes[0]
                while not found_node:
                    for action in actions:
                        if type(action) != int:
                            action_value = action.action
                        else:
                            action_value = action
                        for child in current_node.children:
                            if child.get('action') == action_value:
                                current_node = child.get('node')
                                if np.array_equal(current_node.GameState.state, state):
                                    node = current_node
                                    found_node = True
                                break
                    break
                            
        # Search by node_id                           
        elif node_id is not None:
            while not found_node:
                for tree_node in self.nodes:
                    if tree_node.id == node_id:
                        node = tree_node
                        found_node = True
            pass
        else: 
            print('FIND_NODE does not work - no node_id or state supplied')
                 
        # if node:
            # print('Found Node: ', node.GameState.state)
        return node

    def display_estimated_values(self, node: TreeNode = None, node_id: str = None, state: GameState = None):
        # Set node to display values
        if not node:
            if node_id:
                node = self.find_node(node_id=node_id)
            elif state:
                node = self.find_node(state=state)
            else:
                node = self.nodes[0]
        
        # Create a 3x3 grid for Tic-Tac-Toe board
        board = [['' for _ in range(3)] for _ in range(3)]
        
        # Populate board with truncated UTC values of children, if available
        for child in node.children:
            action = child['action']
            child_node = child['node']
            utc_value = child_node.get_utc_value()  # Assuming each child node has a `utc_value` attribute
            count = child_node.n_visits
            
            # Map action to board coordinates (row, col)
            row, col = divmod(action, 3)
            
            # Format UTC value and visits count to fit neatly
            truncated_value = f"{utc_value:.3g}"  # Truncate to 3 significant figures
            board[row][col] = f"{truncated_value} - {count}"
        
        # Display board
        print("Tic-Tac-Toe Board (UTC Values):")
        for row in board:
            print(" | ".join(value if value else " " for value in row))
            print("-" * 13)


class MCTS:
    """
    This class creates an agent that is an implementation of the Monte Carlo Tree Search algorithm to play tic-tac-toe
    
    ATTRIBUTES:
        - env: The game environment from OpenAI Gym.
        - root_node (TreeNode): The root node of the search tree.
        - tree_graph (TreeGraph): The tree structure used to store nodes.
        - C (int): The exploration term used in the UCT formula.
        - games_completed (int): The number of games completed during the search.

    METHODS:
        - __init__(): Initializes the MCTS with the given environment name and exploration term.
        - selection(): Selects the best child node (based off UTC score) to explore based on UCT values.
        - expansion(): Expands a node by adding all possible child nodes which represent future game states.
        - backpropagation(): Updates the node values along the path after a simulation.
        - think(): Runs the MCTS algorithm. (This can be seen as the Simulation part of the MCTS algorithm)
    """
    def __init__(self, env_name: str, exploration_term: int):
        self.env = gym.make(env_name)
        self.root_node = TreeNode(self.env.reset())
        self.tree_graph = TreeGraph()
        self.tree_graph.add_node(self.root_node)
        self.C = exploration_term
        self.games_completed = 0
        
    def selection(self, node: TreeNode,  player_turn: int, find_best: bool = True, verbose: bool = False):
                
        action_mappings = []
        
        if not find_best:
            
            parent_n_visit = node.n_visits
                
            for item in node.children:
                
                child_node: TreeNode = item['node']
                action = item['action']
                    
                utc_value = child_node.get_utc_value(parent_visits=parent_n_visit, exploration_term=self.C, verbose=False)
                
                action_mapping = Action(action, player_turn, child_node.id, utc_value)
                action_mappings.append(action_mapping)
        else:
            for item in node.children:
                child_node = item['node']
                action = item['action']
                action_mapping = Action(action, player_turn, child_node.id, child_node.value)
                action_mappings.append(action_mapping)
                        
        # Filter for only available moves
        action_mask = node.get_action_mask()
        if len(action_mappings) > 0:
            action_mappings = [action_mapping for action_mapping in action_mappings if action_mapping.action in action_mask]
        else:
            print('FAILED TO CREATE VALID MOVES')

        best_child = max(action_mappings, key=lambda x: x.utc_value)
        if verbose:
            if node == self.tree_graph.root_node:
                print('+' * 10)
                print('- ' * 10, 'SELECTION FUNC()', '- ' * 10)
                print('Action = UtcValue')
                for action in action_mappings:
                    temp_node: TreeNode = self.tree_graph.find_node(node_id=action.tree_node_id)
                    print(f"{action.action} ({action.utc_value}) - [{temp_node.n_visits}, {temp_node.total_reward}]")
                    print('+' * 10)
                    
                print(f'## Best Child ## {best_child.action}')
                
        return best_child
    
    def expansion(self, node: TreeNode, env_ = None):
        for action in node.get_action_mask():   
            if not env_:
                env = deepcopy(self.env)
            else:
                env = deepcopy(env_)
            tree_node = TreeNode(env.step(action), node)
            node.children.append({'node': tree_node, 'action': action})
            self.tree_graph.add_node(tree_node)
            node.is_expanded = True
        
    def backpropagation(self, state: GameState, all_actions: list, backpropagation_actions: list, 
                        reset_env: bool = False, verbose: bool = False):
        
        if verbose:
            print('-'*25)
            print('Backpropagation Summary')
            print('state: ', state)
            
        winner = state.winner
        
        for action in all_actions:
            child_node: TreeNode = self.tree_graph.find_node(node_id=action.tree_node_id)
            node = child_node.parent
            if action.player_turn == winner:
                reward = -1
            elif action.player_turn == -winner:
                reward = 1
            elif winner == 0:
                reward = 0
                
            if action in backpropagation_actions:
                node.update_value(reward)
            
                if verbose:
                    print('<>'*25)
                    print(node)
                    print(reward)
                    print('<>'*25)
        
        if reset_env:
            self.env.reset()
                        
    def think(self, state: GameState, time_allowed: int = 30, previous_actions: list = None, find_best: bool = False, update_values: bool = True, verbose: bool = False):

        start_time = datetime.now()
        original_state = deepcopy(state)
                
        # Summary Statistics
        local_games_completed = 0
        
        if not find_best:
            while (datetime.now() - start_time).total_seconds() < time_allowed:
                
                is_terminated = False
                env_ = deepcopy(self.env)
                state = original_state
                local_games_completed += 1
                
                all_actions = []
                backpropagaion_actions = []
                if previous_actions:
                    all_actions.extend(previous_actions)
                
                while not is_terminated:
                    
                    # Get the TreeNode that this game represents
                    node = self.tree_graph.find_node(state=state.state, actions=all_actions)
                    
                    if not node.is_expanded:
                        self.expansion(node, env_)
                        
                    # Let MCTS algo pick 'best' action
                    best_action: Action = self.selection(node, state.player_turn, find_best=False, verbose=False)
                    
                    all_actions.append(best_action)
                    backpropagaion_actions.append(best_action)
                                                                        
                    state = GameState(env_.step(best_action.action))
                                                            
                    if state.is_terminal:
                        is_terminated = True
            
                if update_values:
                    self.games_completed += 1
                    self.backpropagation(state=state, all_actions=all_actions, backpropagation_actions=backpropagaion_actions,
                                         reset_env=False, verbose=False)
                                        
        # Pick the best option
        node = self.tree_graph.find_node(state=original_state.state, actions=previous_actions)
        best_action = self.selection(node, state.player_turn, find_best=True, verbose=False)
        state = self.env.step(best_action.action)
        
        if verbose:
            print(f'thinking summary: \n - games completed: {local_games_completed}')
            print(self.tree_graph.root_node.n_visits)
            # self.tree_graph.display_estimated_values(node=node)
            
            
        return best_action


class RandomAgent:
    """
    This class represents a random agent that selects moves randomly from the available action space in Tic-Tac-Toe.

    ATTRIBUTES:
        - name (str): The name of the agent.

    METHODS:
        - __init__(): Initializes the RandomAgent with a default name.
        - think(): Selects a random action from the available action space and returns it as an Action object.
    """
    def __init__(self):
        self.name = 'randy'
    
    def think(self, action_space: list):
        action = Action(choice(action_space), 0, 'N/A', 0)
        return action
      

class TrainingEnv:
    """
    This class provides a training environment for the MCTS agent, allowing different training options and configurations.

    ATTRIBUTES:
        - env: The game environment created using OpenAI Gym.
        - thinking_time (int): The allowed thinking time for the MCTS agent during simulations.
        - training_iterations (int): The number of training iterations to perform.
        - games_against_agent (int): The number of games to play against a random agent for evaluation.
        - agent_info (list): A list containing information about the MCTS and Random agents.
        - results (list): Stores the results from training and evaluation.

    METHODS:
        - __init__(): Initializes the TrainingEnv with the specified parameters and loads the agents.
        - load_agents(): Loads the MCTS and Random agents into the environment.
        - self_play(): Allows the MCTS agent to play against itself for training purposes.
        - play_agent(): Makes the MCTS agent play against a random agent for evaluation.
        - update_config(): Updates the training iterations and thinking time configurations.
        - train_agent(): Iteratively trains the agent and evaluates its performance after each iteration.
        - visualise_results(): Visualizes the training and evaluation results.
    """
    def __init__(self, env_name: str, training_iterations: int, exploration_term: int, thinking_time: int, games_against_agent: int = 10):
        self.env = gym.make(env_name)
        self.thinking_time = thinking_time
        self.training_iterations = training_iterations
        self.games_against_agent = games_against_agent
        self.training_logger = Logger('training_logger', 'logs/training.log', console_output=True).get_logger()
        self.testing_logger = Logger('testing_logger', 'logs/testing_logger.log', console_output=False).get_logger()
        self.load_agents(env_name, exploration_term)
        self.results = []
        
    def load_agents(self, env_name, exploration_term):
                
        self.agent_info = [
            {'name': 'MctsAgent', 'agent': MCTS(env_name, exploration_term), 'vs_games_played': 0,
             'games_won': 0, 'games_drawn': 0, 'games_lost': 0},
            {'name': 'RandomAgent', 'agent': RandomAgent(), 'games_played': 0, 'games_won': 0,
             'games_drawn': 0, 'games_lost': 0}
        ]
        
    def self_play(self, verbose: bool = False, training_iterations: str = 1):
                
        for i in range(training_iterations):
            
            if verbose:
                self.training_logger.info('[]'*25)
                self.training_logger.info(f'[] Game: {i}')
                self.training_logger.info('[]'*25)
            
            state = GameState(self.env.reset())
            is_terminated = False
            previous_actions = []
            agent_info = next((agent for agent in self.agent_info if agent['name'] == 'MctsAgent'), None)
            agent: MCTS = agent_info.get('agent')
            agent.env.reset()

            
            while not is_terminated:
                
                action = agent.think(state, time_allowed=self.thinking_time, previous_actions=previous_actions, verbose=False)
                
                # action = self.MCTS.think(state, time_allowed=self.thinking_time, previous_actions=previous_actions, verbose=False)
                previous_actions.append(action)
                state = GameState(self.env.step(action.action))
                
                if verbose:
                    self.training_logger.info('-'*25)
                    self.training_logger.info(f'[] ACTION TAKEN: {action.action} {state}')
                    self.training_logger.info('-'*25)
                
                if state.is_terminal:
                    is_terminated = True
                            
            if verbose:
                self.training_logger.info('[]'*25)
                self.training_logger.info('SELF-PLAY GAME COMPLETE')
                self.training_logger.info(f'FULL GAMES EXPLORED: {agent.games_completed}')
                print(agent.tree_graph.display_estimated_values())
                self.training_logger.info(f'WINNER: {state.winner}')
                self.training_logger.info('[]'*25)

        self.MCTS = agent
        
    def play_agent(self, games_to_play: int = 50, verbose: bool = False, starting_player_option: str = 'RANDOM'):
        """
        This will make the MCTS Agent play against an agent which randomly selects a move from the available action space
        """
        players = [
            {'player': 'MCTS', 'wins': 0, 'draws': 0, 'losses': 0},
        ]
        player_names = ['MCTS', 'random']
        self.RandomAgent = RandomAgent()
        self.MCTS.env.reset()
        
        for i in range(games_to_play):
            state = GameState(self.env.reset())
            is_terminated = False
            previous_actions = []
            
            if starting_player_option == 'RANDOM':
                starting_player = choice(player_names)
            elif starting_player_option == 'MCTS':
                starting_player = player_names[0]
            elif starting_player_option == 'RANDOM_AGENT':
                starting_player = player_names[1]
                
            player_turn = starting_player
            self.testing_logger.info('[]'*25)
            self.testing_logger.info(f'TESTING GAME: {i}')
            self.testing_logger.info(f'Starting player: {starting_player} - {starting_player_option}')
            self.testing_logger.info('[]'*25)

            
            while not is_terminated:
                
                if player_turn == 'MCTS':
                    action = self.MCTS.think(state=state, time_allowed=self.thinking_time, previous_actions=previous_actions, update_values=False)
                else:
                    node = self.MCTS.tree_graph.find_node(state=state.state, actions=previous_actions, verbose=False)
                    action_space = node.get_action_mask()
                    action = self.RandomAgent.think(action_space=action_space)
                    self.MCTS.env.step(action.action)
                previous_actions.append(action)
                
                state = GameState(self.env.step(action.action))
                
                self.testing_logger.info('-'*25)
                self.testing_logger.info(f'[{player_turn}] action taken: {action.action} \n \n {state}')
                self.testing_logger.info('-'*25)
                    
                player_turn = [player for player in player_names if player != player_turn][0]
                
                if state.is_terminal:
                    if verbose:
                        if state.winner == 1:
                            winner_name = starting_player
                        elif state.winner == -1:
                            winner_name = [player for player in player_names if player != starting_player][0]
                        else:
                            winner_name = 'DRAW'
                        self.testing_logger.info('~'*25)
                        self.testing_logger.info(f'Winner: {winner_name}')
                        self.testing_logger.info('~'*25)
                        print('~'*25)
                        print(f'~ Winner: {winner_name}')
                        print('~'*25)
                    is_terminated = True
                    self.MCTS.env.reset()
                    
            if state.winner == 1:
                winner = starting_player
            elif state.winner == -1:
                winner = [player for player in player_names if player != starting_player][0]
            elif state.winner == 0:
                winner = 'DRAW'
                
            for player_dict in players:
                if winner == player_dict.get('player'):
                    player_dict['wins'] = player_dict['wins'] + 1
                elif winner == 0:
                    player_dict['draws'] = player_dict['draws'] + 1
                else: 
                    player_dict['losses'] = player_dict['losses'] + 1
            
        return players
    
    def update_config(self, training_iterations: int, thinking_time: int):
        if training_iterations:
            self.training_iterations = training_iterations
        if thinking_time: 
            self.thinking_time = thinking_time
       
    def train_agent(self, verbose: bool = True):
        """
        This is a function that will iteratively train the agent while stopping its training process after each
        iteration to make it play games against a random agent which will allow us to assess the progression
        during the training process
        """
        
        for _ in range(self.training_iterations):
            results = {'games_completed': 0, 'starting_results': None, 'second_results': None}
            for training_index in range(10):
                self.self_play(verbose=verbose, training_iterations=1)
            results['games_completed'] = self.MCTS.games_completed
            
            player_results_when_starting = self.play_agent(verbose=verbose, games_to_play=self.games_against_agent, starting_player_option='MCTS')
            results['starting_results'] = player_results_when_starting
            player_results_when_second = self.play_agent(verbose=verbose, games_to_play=self.games_against_agent, starting_player_option='RANDOM_AGENT')
            results['second_results'] = player_results_when_second
            self.results.append(results)
        self.visualise_results()
    
    def visualise_results(self):

        # Build a list of dictionaries for the data
        data_list = []

        for result in self.results:
            completed_games = result['games_completed']

            # Starting results
            starting_result = result['starting_results'][0]
            s_wins = starting_result['wins']
            s_draws = starting_result['draws']
            s_losses = starting_result['losses']
            s_total = s_wins + s_draws + s_losses
            s_win_rate = s_wins / s_total if s_total > 0 else 0

            # Second results
            second_result = result['second_results'][0]
            sec_wins = second_result['wins']
            sec_draws = second_result['draws']
            sec_losses = second_result['losses']
            sec_total = sec_wins + sec_draws + sec_losses
            sec_win_rate = sec_wins / sec_total if sec_total > 0 else 0

            data_list.append({
                'games_completed': completed_games,
                'starting_wins': s_wins,
                'starting_draws': s_draws,
                'starting_losses': s_losses,
                'starting_win_rate': s_win_rate,
                'second_wins': sec_wins,
                'second_draws': sec_draws,
                'second_losses': sec_losses,
                'second_win_rate': sec_win_rate
            })

        df = pd.DataFrame(data_list)
        df = df.sort_values(by='games_completed').reset_index(drop=True)

        # Visualization 1: Line Plot of Win Rates Over Games Completed
        plt.figure(figsize=(10, 6))
        plt.plot(df['games_completed'], df['starting_win_rate'], label='MCTS Starting First', marker='o')
        plt.plot(df['games_completed'], df['second_win_rate'], label='MCTS Playing Second', marker='o')
        plt.xlabel('Games Completed')
        plt.ylabel('Win Rate')
        plt.title('MCTS Win Rate Over Games Completed')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Visualization 2: Grouped Bar Chart of Wins, Draws, and Losses
        ind = np.arange(len(df))  # the x locations for the groups
        width = 0.35  # the width of the bars

        plt.figure(figsize=(12, 6))

        # Bars for MCTS Starting First
        plt.bar(ind - width/2, df['starting_wins'], width/3, label='Wins (Starting First)', color='green')
        plt.bar(ind - width/2 + width/3, df['starting_draws'], width/3, label='Draws (Starting First)', color='blue')
        plt.bar(ind - width/2 + 2*(width/3), df['starting_losses'], width/3, label='Losses (Starting First)', color='red')

        # Bars for MCTS Playing Second
        plt.bar(ind + width/2, df['second_wins'], width/3, label='Wins (Playing Second)', color='lightgreen')
        plt.bar(ind + width/2 + width/3, df['second_draws'], width/3, label='Draws (Playing Second)', color='cyan')
        plt.bar(ind + width/2 + 2*(width/3), df['second_losses'], width/3, label='Losses (Playing Second)', color='salmon')

        plt.xlabel('Games Completed')
        plt.ylabel('Number of Games')
        plt.title('MCTS Wins, Draws, and Losses Over Games Completed')
        plt.xticks(ind, df['games_completed'])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


# Define TrainingEnv configuration
config = {
    'env_name': 'tictactoe-v0',
    'training_iterations': 10,
    'exploration_term': 2,
    'thinking_time': 2,
    'games_against_agent': 50
}

# Setup the TrainingEnv and begin training agent
TrainingEnv = TrainingEnv(**config)
TrainingEnv.train_agent(verbose=True)

    