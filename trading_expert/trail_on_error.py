import random

from expert.basic_expert import BaseAbility, Action
from trading import TradeAction


class TrailOnErrorDecision(BaseAbility):

    def act(self, state) -> Action:
        decision_val = random.randint(0, 1000) % 3
        action = Action.action_gen(TradeAction.STAY, 100)
        if decision_val == 0:
            action = Action.action_gen(TradeAction.SELL, 90)
        elif decision_val == 2:
            action = Action.action_gen(TradeAction.BUY, 90)
        return action

    def evaluate(self, state, result):
        # print(f"evaluate {state, result}")
        pass
