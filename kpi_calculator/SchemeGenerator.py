import time
import pandas as pd
import datetime

import numpy as np
import matplotlib.pyplot as plt

from var_dump import var_dump
from climatics_client.retrieve import Retriever

import requests

from kpi_calculator.KPICalculator import KPICalculator


class SchemeGenerator:
    def __init__(self) -> None:
        self.DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']


    def generate_scheme(self, df):
        # We specify the days of a week because the api returns the days of the week out of order
        kpi_calculator = KPICalculator()
        scheme = []
        # df.to_csv('out.csv')
        for day in self.DAYS:
            best_score = 0
            best_comfort_timespan = []

            for index in range(len(df.loc[day])):
                if df.loc[day].iloc[index].occupancy > 0.5:
                    first_non_zero_value_index = round(index/5)
                    break

            for index in range(len(df.loc[day]) - 1, -1, -1):
                if df.loc[day].iloc[index].occupancy > 0.5:
                    last_non_zero_value_index = round(index/5)
                    break

            # print(first_non_zero_value_index, last_non_zero_value_index)
            for start in range(0, len(df.loc[day]), 5)[first_non_zero_value_index:last_non_zero_value_index]:
                for end in range(0, len(df.loc[day]), 5)[round(start/5):last_non_zero_value_index]:
                    current_score = kpi_calculator.score_generated_scheme(df.loc[day], start, end)
                    if current_score > best_score:
                        best_score = current_score
                        best_comfort_timespan = [start, end]
                        # print(best_score)
                        # print(best_comfort_timespan)
                        # print()
            scheme_day = [best_comfort_timespan[0]]
            scheme_day.append(best_comfort_timespan[1] - best_comfort_timespan[0])
            scheme_day.append(720 - best_comfort_timespan[1])
            scheme.append(scheme_day)

            print(day, best_score, best_comfort_timespan)
        self.chart_scheme_and_occupancy(scheme, df)

    def chart_scheme_and_occupancy(self, scheme, df):
        print(scheme)
        weekly_occupancy = []
        top_line = [1] * 5040
        bottom_line = [0] * 5040
        format_line = ([1, 0] + [.5] * 718) * 7
        scheme_line = []
        for scheme_day in scheme:
            scheme_line.extend([0] * scheme_day[0]
                             + [1] * scheme_day[1]
                             + [0] * scheme_day[2])



        for day in self.DAYS:
            weekly_occupancy.extend(df.loc[day]['occupancy'].tolist())
        
        plt.plot(format_line, 'red', linewidth = .5)
        plt.plot(weekly_occupancy, 'green')
        plt.fill_between(range(5040), scheme_line, bottom_line, color = 'yellow', alpha = .5)
        plt.fill_between(range(5040), scheme_line, top_line, color = 'blue', alpha = .3)
        tick_names = [[day[:3], '6:00', '12:00', '18:00'] for day in self.DAYS]
        tick_names = [item for sublist in tick_names for item in sublist]
        ticks = [x * 180 for x in range(28)]
        plt.xticks(ticks, tick_names)
        plt.yticks([x / 10 for x in range(11)], [str(x * 10) + '%' for x in range(11)])
        plt.ylabel('Bezettingsgraad')
        plt.xlabel('Timestamp')
        plt.grid()
        plt.show()



    def create_visual_schedule(self, scheme):
        print(scheme)
        df = pd.DataFrame(scheme, columns=['Day', 'Eco', 'Comfort', ''])
        # view data
        print(df)
        
        # plot data in stack manner of bar type
        df.plot(x='Day', kind='bar', stacked=True,
                title='Scheme')
        plt.show()






