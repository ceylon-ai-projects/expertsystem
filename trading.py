from enum import unique

from expert.basic_expert import ActionType


@unique
class TradeAction(ActionType):
    BUY = 1
    STAY = 2
    SELL = 3
