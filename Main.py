import time

import numpy as np
import os

from data.data_manager import get_data_chunk
from expert.basic_expert import Expert, Action
from trading import TradeAction
from trading_expert.trail_on_error import TrailOnErrorDecision

display = True


class TradingExpert(Expert):

    def after_init(self):
        self.abilities.append(TrailOnErrorDecision())


tradingExpert = TradingExpert(strike_step=3)

pair_name = "EURUSD"
interval = "1.mini"

dirname = os.path.dirname(__file__)
base_path = dirname + ""

df_csv = get_data_chunk(pair_name, interval, 2048)

risk_factor = 0.0002


def calculate_certainty(diff):
    certainty = 0.0
    decision_val = abs(diff) / risk_factor
    # print(decision_val)
    if decision_val >= 3:
        certainty = 1
    elif decision_val >= 2:
        certainty = 0.8
    elif decision_val >= 1:
        certainty = 0.6
    else:
        certainty = 0.4

    return certainty


def calculate_result(__states):
    # print(states)
    f_step = __states[:1, 4:5]  # using close value
    l_step = __states[-1:, 4:5]  # using close value

    diff = f_step - l_step
    if diff == 0:
        certainty = 1
        action_type = TradeAction.STAY
    else:
        certainty = calculate_certainty(diff)
        if diff > 0:
            action_type = TradeAction.SELL
        else:
            action_type = TradeAction.BUY

    __action = Action.action_gen(action_type, certainty)

    # print(f_step, l_step, diff, diff > 0, calculate_certainty(diff))
    # print(__action)
    # time.sleep(0.5)

    return __action


if __name__ == '__main__':
    print("Expert working")
    last_result = -1

    for df in df_csv:
        step_count = 0

        next_strike = 0
        states = []
        print(df.head())
        print(tradingExpert.status())

        for df_row in df.values:
            df_values = df_row

            states.append(df_values)

            if step_count == next_strike:
                states = np.array(states)
                result = calculate_result(states)

                # tradingExpert.feedback(result=result, state=states)
                action, strike_on = tradingExpert.interact(states, result)

                next_strike = step_count + strike_on
                act_state = df_values
                next_action = action
                ## States after action taken
                states = []
                # states.append(df_values)

            step_count += 1
        if display:
            time.sleep(0.2)
