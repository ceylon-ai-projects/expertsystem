from argparse import Action
from enum import Enum


class ActionType(Enum):
    pass


class Action:
    type = None
    certainty = 0.0

    def __init__(self, _type, _certainty) -> None:
        self.certainty = _certainty
        self.type = _type

    @staticmethod
    def action_gen(_type, certainty) -> Action:
        return Action(_type, certainty)

    def __str__(self) -> str:
        return f"( {self.type} - {self.certainty} )"

    def __repr__(self):
        return self.__str__()


class BaseAbility:
    last_action = None

    def act(self, state,result) -> Action:
        return None

    def evaluate(self, state, result):
        pass

    def status(self):
        pass


class Expert:
    abilities = []
    last_state = None
    strike_step = 1

    def __init__(self, strike_step=1):
        self.strike_step = strike_step
        self.after_init()

    def after_init(self):
        pass

    def interact(self, state, result):
        actions = []
        for ability in self.abilities:
            action = ability.act(state, result)
            actions.append(action)
        self.last_state = state
        return actions, self.strike_step

    def feedback(self, result, state=None):
        state = self.last_state if state is None else state
        for ability in self.abilities:
            ability.evaluate(state, result)

    def status(self):
        print("--Expert System Summary--\n\n")
        print(f"Total abilities {len(self.abilities)}")
        for ability in self.abilities:
            ability.status()
