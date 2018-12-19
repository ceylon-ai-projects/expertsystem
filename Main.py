import random
from enum import unique

from expert.basic_expert import Expert, BaseAbility, Action, ActionType


@unique
class TradeAction(ActionType):
    BUY = 1
    STAY = 2
    SELL = 3


class RandomDecision(BaseAbility):

    def act(self, state) -> Action:
        decision_val = random.randint(0, 1000) % 3
        action = Action.action_gen(TradeAction.STAY, 100)
        if decision_val == 0:
            action = Action.action_gen(TradeAction.SELL, 90)
        elif decision_val == 2:
            action = Action.action_gen(TradeAction.BUY, 90)
        return action

    def evaluate(self, state, result):
        print(f"evaluate {state, result}")


class TradingExpert(Expert):

    def __init__(self):
        self.abilities.append(RandomDecision())


tradingExpert = TradingExpert()

if __name__ == '__main__':
    print("Expert working")
    last_result = -1
    for i in range(100):
        action = tradingExpert.interact(i)
        print(action)
        if last_result != -1:
            tradingExpert.feedback(i % 3 == 0, last_result)
        last_result = i
