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

    def act(self, state) -> Action:
        return None

    def evaluate(self, state, result):
        pass


class Expert:
    abilities = []

    last_state = None

    def interact(self, state):
        actions = []
        for ability in self.abilities:
            action = ability.act(state)
            actions.append(action)
        self.last_state = state
        return actions

    def feedback(self, result, state=None):
        state = self.last_state if state is None else state
        for ability in self.abilities:
            ability.evaluate(state, result)
