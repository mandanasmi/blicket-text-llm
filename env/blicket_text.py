"""
Simple Text Environment Module
"""
import copy 
import re
from collections import defaultdict

import numpy as np


class BlicketFunctionSet:
    """Static class, set of functions for blicket text env"""
    
    @staticmethod
    def conjunctive(item_states: np.ndarray, blicket_indices: np.ndarray):
        """True if all blickets on machine"""
        blicket_states = item_states[blicket_indices]
        return np.all(blicket_states) 

    @staticmethod
    def disjunctive(item_states: np.ndarray, blicket_indices: np.ndarray):
        """True if any blicket on machine"""
        blicket_states = item_states[blicket_indices]
        return np.any(blicket_states)

    @staticmethod
    def get_function(name: str):
        return getattr(BlicketFunctionSet, name)
    

ENV_DESC = "You are in a room. You see a machine at the center of this room. "\
"\n\nThere are also #NUM_OBJECT# objects scattered around the room. #OBJECT_DESC# "\
"\n\nThe machine hums softly in front of you, seemingly waiting. #MACHINE_STATE# "\
"You wonder if there is a relationship between the objects and the machine."


class BlicketTextEnv:
    def __init__(self, num_objects: int, num_blickets: int, init_prob: float, 
                 transition_noise: float, rule: str,  seed: int = 0):
        self.num_objects = num_objects
        self.num_blickets = num_blickets
        self.init_prob = init_prob # initial probability of object being on machine
        self.transition_noise = transition_noise # noise in machine state transition
        
        self.rule = rule
        self._rule_fn = BlicketFunctionSet.get_function(self.rule)

        self._init_regex_patterns()  # Precompile regex patterns for commands.
        self._rng = np.random.default_rng(seed=seed)

        # 
        self.object_names = [f"object {i}" for i in range(self.num_objects)]  # TODO: make customizable

        # sample blicket indices
        self.blicket_indices = sorted(self._rng.choice(self.num_objects, self.num_blickets, replace=False))

        # _state: first num_objects booleans for objects, last element for machine state.
        self._state = np.empty((self.num_objects + 1,), dtype=bool)
        
        # episode counters
        self._moves_count = 0
        self._state_visits = defaultdict(int)
        self._turn_machine_on = False 
    
    def _init_regex_patterns(self):
        # Allow either "machine"/"floor" or "the machine"/"the floor".
        self.put_regex = re.compile(
            r"put\s+(.+?)\s+on\s+(?:the\s+)?(machine|floor)$", re.IGNORECASE
        )
        # Allow "off machine" or "off the machine", with an optional "of"
        self.take_regex = re.compile(
            r"take\s+(.+?)\s+off\s+(?:of\s+)?(?:the\s+)?machine$", re.IGNORECASE
        )
        self.look_regex = re.compile(r"look(\s+around)?$", re.IGNORECASE)
        self.exit_regex = re.compile(r"exit$", re.IGNORECASE)
    
    def _update_machine_state(self):
        """Update machine state based on current object states"""
        machine_state = self._rule_fn(self._state, self.blicket_indices)
        if self._rng.random() < self.transition_noise:
            machine_state = not machine_state  # random flip
        self._state[-1] = machine_state
        return machine_state

    def _get_object_description(self):
        descriptions = []
        for i in range(self.num_objects):
            loc = "on top of the machine" if self._state[i] else "on the floor"
            descriptions.append(f"{self.object_names[i]} is {loc}")
        return "You observe them: " + ", ".join(descriptions) + "."
    
    def _get_machine_description(self):
        light_state = "now on" if self._state[-1] else "currently off"
        return "The light on the machine is " + light_state + "."
    
    def reset(self):
        self._state[:-1] = self._rng.random(self.num_objects) < self.init_prob
        self._update_machine_state()

        self._state_visits = defaultdict(int)
        self._state_visits[hash(self._state.tobytes())] += 1  # unique state visits

        self._turn_machine_on = False 
        self._moves_count = 0

        feedback = ENV_DESC.replace("#NUM_OBJECT#", str(self.num_objects))
        feedback = feedback.replace("#OBJECT_DESC#", self._get_object_description())
        feedback = feedback.replace("#MACHINE_STATE#", self._get_machine_description())
        
        return {
            "feedback": feedback,
            "true_state": np.copy(self._state),
            "object_names": copy.deepcopy(self.object_names),
            "blicket_indices": copy.deepcopy(self.blicket_indices),
            "moves": self._moves_count,
            "unique_state_visited": len(self._state_visits),
            "turn_machine_on": self._turn_machine_on,
        }
    
    def step(self, command: str):
        command = command.lower().strip()
        feedback = ""
        reward = 0.0
        done = False
        update_machine = False  # only update machine state if put/take command

        if self.look_regex.fullmatch(command):
            feedback = ENV_DESC.replace("#NUM_OBJECT#", str(self.num_objects))
            feedback = feedback.replace("#OBJECT_DESC#", self._get_object_description())
            feedback = feedback.replace("#MACHINE_STATE#", self._get_machine_description()) 
        elif self.exit_regex.fullmatch(command):
            feedback = "Exiting the episode."
            done = True
        elif (match := self.put_regex.fullmatch(command)):
            object_name, destination = match.group(1), match.group(2)
            feedback = self._handle_put(object_name, destination)
            update_machine = True
        elif (match := self.take_regex.fullmatch(command)):
            object_name = match.group(1)
            feedback = self._handle_take(object_name)
            update_machine = True
        else:
            feedback = ("Invalid command. Valid commands are: 'put <object> on [the] machine', "
                        "'put <object> on [the] floor', 'take <object> off [of] [the] machine', "
                        "'look', or 'exit'.")
        
        if update_machine:
            self._update_machine_state()
        if not done:
            feedback = feedback + " " + self._get_machine_description()

        self._state_visits[hash(self._state.tobytes())] += 1  # unique state visits
        self._moves_count += 1
        self._turn_machine_on = self._turn_machine_on or self._state[-1]

        return {
            "feedback": feedback,
            "true_state": np.copy(self._state),
            "object_names": copy.deepcopy(self.object_names),
            "blicket_indices": copy.deepcopy(self.blicket_indices),
            "moves": self._moves_count,
            "unique_state_visited": len(self._state_visits),
            "turn_machine_on": self._turn_machine_on,
        }, reward, done
    
    def _handle_put(self, object_name: str, destination: str) -> str:
        if object_name not in self.object_names:
            return f"{object_name} is not a recognized object."
        
        idx = self.object_names.index(object_name)
        desired_state = (destination == "machine")
        current_location = "machine" if self._state[idx] else "floor"
        if self._state[idx] == desired_state:
            return f"{object_name} is already on the {current_location}."
        
        self._state[idx] = desired_state
        if desired_state:
            return f"You put {object_name} on the machine."
        else:
            return f"You put {object_name} on the floor."
    
    def _handle_take(self, object_name: str) -> str:
        if object_name not in self.object_names:
            return f"{object_name} is not a recognized object."
        
        idx = self.object_names.index(object_name)
        if not self._state[idx]:
            return f"{object_name} is already on the floor."
        
        self._state[idx] = False
        return f"You took {object_name} off of the machine."


if __name__ == "__main__":
    env = BlicketTextEnv(num_objects=5, num_blickets=1, init_prob=0.1, transition_noise=0.01, rule="conjunctive")
    print(f"True blicket indices: {env.blicket_indices}")
    game_state = env.reset()
    print(game_state['feedback'])
    while True:
        action = input("Enter command: ")
        game_state, reward, done = env.step(action)
        print(game_state['feedback'])
        if done:
            break
    # print("Game over!")