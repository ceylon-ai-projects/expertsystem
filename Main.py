import time
import os
import numpy as np
from data.data_manager import get_data_chunk
from expert.basic_expert import Expert, Action
from trading import TradeAction
from trading_expert.trail_on_error import TrailOnErrorDecision


class TradingExpert(Expert):

    def __init__(self):
        self.abilities.append(TrailOnErrorDecision())


tradingExpert = TradingExpert()

pair_name = "EURUSD"
interval = "1.mini"

dirname = os.path.dirname(__file__)
base_path = dirname + ""

df_csv = get_data_chunk(pair_name, interval, 1000)


def calculate_result(states, action):
    print(states[:1], states[-1:])
    print(action)

    return Action.action_gen(TradeAction.STAY, 100)


if __name__ == '__main__':
    print("Expert working")
    last_result = -1

    for df in df_csv:
        step_count = 0

        next_action = None
        next_strike = -1
        act_state = None
        states = []
        print(df.head())

        for df_row in df.values:
            df_values = df_row

            states.append(df_values)

            if step_count == next_strike:
                states = np.array(states)
                result = calculate_result(states, next_action)
                tradingExpert.feedback(result=result, state=states)

            if next_strike <= step_count:
                action, strike_on = tradingExpert.interact(df_values)
                next_strike = step_count + strike_on
                act_state = df_values
                next_action = action
                states = []
                states.append(df_values)

            step_count += 1
            time.sleep(0.5)
