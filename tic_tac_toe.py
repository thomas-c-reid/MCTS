import numpy as np
import gym
import tictactoe_gym
import copy
import time
from random import choice
from uuid import uuid4
from gym.envs.registration import register
from dataclasses import dataclass


register(
    id='tictactoe-v0',
    entry_point='tictactoe_gym.envs.tictactoe_env:TicTacToeEnv',
)

"""
This is a file where I will create a MCTS algorithm to play tic-tac-toe

Classes

1) TicTacToeState:
    - contains all the info on current state of the game
    
2) TreeNode:
    - Contains information relating the the current TicTacToeState which will be part of the tree graph
    - Contains information which can help MCTS algorithm deduct if a state is 'promising'
    
3) PolicyNetwork
    - NOT STARTED
    - This class will hold the NN which is responsible for outputting an initial 'rating' for each game state
    - Will need trained over time
    - should be trained in batches
    - MAIN PROBLEM: I am not sure how the policys output value for a state should factor into the entire MCTS algorithms final calculation
    
4) TreeGraph
    - stores all Node representation of states
    - Has helper functions for navigating tree and adding to it
    
5) MCTS
    - Still to be programmed
    
    
TODO:
we want for it to 
"""

class TicTacToeState:
    
    def __init__(self, observation_space):
        if type(observation_space) == tuple:
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
        if self.state is None:
            x = 1
        state_str = '\n'.join([' '.join(map(str, row)) for row in self.state])
        return (
            f"{'#' * 25}\n"
            f"Game State:\n"
            f"{'#' * 25}\n\n"
            f"State:\n"
            f"{state_str}\n\n"
            f"Player Turn  : {self.player_turn}\n"
            f"Is Terminal  : {self.is_terminal}\n"
            f"Winner       : {self.winner}\n"
            f"{'#' * 25}\n"
        )
       
class TreeNode:
        
    def __init__(self, observation_space=None, parent=None, previous_action=None):
        self.id = uuid4()
        self.number_of_visits = 0
        self.total_reward = 0
        self.value = 0
        self.previous_action = previous_action
        self.tictactoeState = TicTacToeState(observation_space=observation_space)
        self.policy_value = 0
        self.parent = parent
        self.is_expanded = False
        self.action_mapping = []
        
        self.children = []
        
    def update_value(self, reward):
        """
        New value derived from average reward per visit
        """
        self.number_of_visits += 1
        self.total_reward += reward
        self.value = reward / self.number_of_visits
   
   # Improve this 
    def __str__(self):
        output = f'State: \n {self.tictactoeState} \n Value: {self.value}'
        return output
         
class PolicyNetwork:
    
    def __init__(self):
        '''
        Based on input of a certain size, output the value for the current state
        After each simulation, replay buffer updated with previous experiences        
        '''
        self.replay_buffer = []
        pass
    
    def forward(self, state, action_space_mask):
        """
        preform a forward pass through the policy network
        """
        pass
    
    def update(self, state: TicTacToeState, ):
        """
        FUNCTION TO UPDATE NETWORK
        
        We will preform forward pass using previous experience and use target probabilites which are derived from each child nodes values after each simulation  
        """

class TreeGraph:
    """
        This is a class to keep track of the different tree nodes and return a certain tree node based off a state
    """
    
    def __init__(self):
        self.nodes = []
        
    def add_node(self, node: TreeNode):
        self.nodes.append(node)
        
    def find_node(self, state: TicTacToeState = None, id: uuid4 = None):
        node = None
        if state:
        # you want to compare the state with each nodes state
            found_node = False
            while not found_node:
                for tree_node in self.nodes:
                    if np.array_equal(tree_node.tictactoeState.state, state.state):
                        node = tree_node
                        found_node = True
                break
        elif id:
            found_node = False
            while not found_node:
                for tree_node in self.nodes:
                    if tree_node.id == id:
                        node = tree_node
                        found_node = True
                break
                    
        return node
        
class MCTS:
    """
    MCTS ALGORITHM
    
    SELECTION:
    1) First the MCTS selects a child node (next state) which is asssociated with the most promising actions 
        - initially this will be what it considers the optimal move, but after several iterations, it will be a state for explorative reasons
    
    EXPANSION:
    2) Once it selects an unvisited node it uses policy network to output distribution over current moves. A mask will be used to filter out invalid moves
       Search process then continues by picking the UCT value which makes most sense
       
    SIMULATION:
    3) Rest of the game is played out and 
        - need to choose if we want the policy network to guide decisions (this will mean more work), or you can use random moves - with self play, will backpropagate successfully regardless
    
    BACKPROPAGATION:
    4) Simulation completes and outputs reward, this reward is propagated through the tree nodes that were picked. Their value is updated using following formula
        - visit count is incremented
        - value is updated to reflect average outcome of all 
    """
    
    def __init__(self, root_node):
        self.tree_graph = TreeGraph()
        self.tree_graph.add_node(root_node)
        self.root_node = root_node
        self.C = 0.05
        self.official_actions_taken = []
    
    def selection(self, current_node: TreeNode, action_mapping: dict, progress_env_function, form: str = 'exploration'):
        """
        This will use UCT (Upper Confidence Bound For Trees) to pick most promising child node
        """
        
        is_selection_phase_done = False
        actions_taken = []
        
        while is_selection_phase_done:
            if form == 'exploration':
                best_action = None
                
                child_mappings = []
                
                parent_node_total_visits = current_node.number_of_visits
                
                # Calculate UTC values
                for child_node in current_node.children:
                    
                    child_node_total_visits = child_node.number_of_visits
                    if child_node_total_visits == 0:
                        # needs to be added to mapping
                        child_mapping_dict = {
                            'node_id': child_node.id,
                            'value': utc_value
                        }
                        child_mappings.append(child_mapping_dict)
                        continue
                    
                    child_node_total_reward = child_node.total_reward
                    
                    exploitation_value = child_node_total_reward/child_node_total_visits
                    
                    exploration_value = self.C * np.sqrt(np.log(parent_node_total_visits) / child_node_total_visits)
                    
                    utc_value = exploitation_value + exploration_value
                    
                    child_mapping_dict = {
                        'node_id': child_node.id,
                        'value': utc_value
                    }
                    child_mappings.append(child_mapping_dict)
                    
            elif form == 'exploitation':
                # in this case we want to only get the best option, not the UTC - best option
                child_mappings = []
                for child_node in current_node.children:
                    node_dict = {'node_id': child_node.id,'value': child_node.value}
                    child_mappings.append(node_dict) 
                            
            # Pick highest UTC value
            max_utc_score = max(child_mapping['value'] for child_mapping in child_mappings)
            max_utc_score_actions = [child_mapping for child_mapping in child_mappings if child_mapping['value'] == max_utc_score]
            
            if len(max_utc_score_actions) > 1:
                node_id_choose = choice(max_utc_score_actions)
            else:
                node_id_choose = max_utc_score_actions[0]
                
            # relate choice back to action
            node_to_choose: TreeNode = self.tree_graph.find_node(id=node_id_choose.get('node_id'))
            
            for state in action_mapping:
                if np.array_equal(state.get('observation').state, node_to_choose.tictactoeState.state):
                    best_action = state.get('action')   
                    actions_taken.append(best_action)
                    
            # Decide to keep going with Selection Phase or do we need to return
            if node_to_choose.is_expanded or node_to_choose.tictactoeState.is_terminal:
                is_selection_phase_done = True
                
            if not is_selection_phase_done:
                progress_env_function(actions_taken)
                    
            
        return best_action
    
    def expansion(self, current_node: TreeNode, step_function, previous_actions_taken, form='initial_visit'):
        
        if form == 'initial_visit':
            observations = step_function(previous_actions_taken=previous_actions_taken, form='initial_visit')
            current_node.action_mapping = observations
            
            for observation in observations:
                
                state_representation = TicTacToeState(observation.get('observation'))
                            
                node = self.tree_graph.find_node(state=state_representation)
                if node is None:
                    node = TreeNode(observation_space=observation.get('observation'), parent=current_node)
                    self.tree_graph.add_node(node)
                    
                if node not in current_node.children:
                    current_node.children.append(node)
                    
                observation['observation'] = state_representation

        elif form == 'exploration':
            
            observations = step_function(previous_actions_taken=previous_actions_taken, form='exploration')
            
        return observations
        
    def simulation(self):
        # While last_result is not terminal
        # make random_choices until it is terminal
        # return output from env.step when it is terminal
        pass
    
    def backpropgation(self):
        pass
    
    def think(self, current_state: TicTacToeState, step_function: callable, previous_actions_taken: list, time_to_think: int = 30):
        player_turn = current_state.player_turn
        start_time = time.time()
            
        node = self.tree_graph.find_node(state=current_state)
        if not node.is_expanded:
            action_mapping = self.expansion(node, step_function, previous_actions_taken, form='initial_visit')
            
        while time.time() - start_time < time_to_think:
            
            actions_taken = []
            
            # Step 1 - Selection
            best_action = self.selection(node, action_mapping, form='exploration')
            
            # Step 2 - Expansion
            action_mapping = self.expansion(node, step_function, previous_actions_taken=actions_taken, form='exploration')
            
            # Step 3 - simulation
            # after the expansion stage we want to randomly pick moves until the game ends and the env reaches a terminal state
            
            # Step 4 - backpropagation
            
        
        best_action = self.selection(node, action_mapping, form='exploitation')
        
        return best_action
                         
@dataclass
class Action:
    action_taken: int
    players_turn: int
    official_action: bool

class TrainingEnv:
    
    def __init__(self):
        self.env = gym.make('tictactoe-v0')
        self.training_iterations = 1000
        
    def step_function(self, form: str, action=None, previous_actions_taken: list=None):
        """
        This is a helper function which allows the expansion part of the algorithm view the 
        potential states to move to for all valid actions
        """
        observations = []
        if form == 'initial_visit':
            for action in range(self.env.action_space.n):
                env_ = copy.deepcopy(self.env)
                observation = env_.step(action) 
                step_dict = {'action': action, 'observation': observation}
                observations.append(step_dict)
        if form == 'exploration':
            env_ = copy.deepcopy(self.env)
            env_.reset()
            
            
            
        return observations
    
    def progress_env_function(self, actions_taken):
        """
        This is a helper function to progress the env to the next state and return the observation seen at the next state
        """
        env_ = copy.deepcopy(self.env)
        env_.reset()
        for action in actions_taken:
            observation = env_.step(action)
        return observation
    
    def train(self):
        for _ in range(self.training_iterations):
            pass
        
    def start(self):

        self.root_node = TreeNode(observation_space=self.env.reset())
        self.MCTS = MCTS(self.root_node)
        
        for _ in range(self.training_iterations):
            
            state = TicTacToeState(self.env.reset())
            isTerminated = False
            previous_actions_taken = []
            
            while not isTerminated:     
                   
               # currently doesn't work - need a way for it to pick its options
               action = self.MCTS.think(state, self.step_function, previous_actions_taken)
               print('action:', action)
               previous_actions_taken.append(action)
               
               
               
        # action = self.env.action_space.sample()
        # print('Action: ', action)
        observation = self.env.step(action)
        
        new_node = TreeNode(observation, parent=self.root_node)
        self.MCTS.tree_graph.add_node(new_node)
        print(new_node)
                
TrainingEnv = TrainingEnv()
TrainingEnv.start()
