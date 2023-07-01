import datetime
import pandas as pd
import matplotlib.pyplot as plt
from kpi_calculator.KPICalculator import KPICalculator


class SchemeGenerator:
    def __init__(self, occupancy_threshold, kpi_calculator) -> None:
        self.OCCUPANCY_THRESHOLD = occupancy_threshold
        self.DAYS = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        self.kpi_calculator = kpi_calculator


    def generate_scheme(self, df):
        # Check to see if we have enough data to make an entire scheme
        if False in [x in df.index for x in self.DAYS]:
            print('Cannot create a scheme as the given dataframe does not have data for every weekday')
            return False

        total_occupancy_eco, total_occupancy_comfort = [], []
        scheme = []
        for day in self.DAYS:
            if day in df.index:
                best_score = 0
                best_comfort_timespan = [0, 0]
                best_occupancy_eco = []
                best_occupancy_comfort = []
                first_viable_comfort_mode_index = 0
                last_viable_comfort_mode_index = 719

                for index in range(len(df.loc[day])):
                    if df.loc[day].iloc[index].occupancy > self.OCCUPANCY_THRESHOLD:
                        first_viable_comfort_mode_index = round(index/5)
                        break
                
                for index in range(len(df.loc[day]) - 1, -1, -1):
                    if df.loc[day].iloc[index].occupancy > self.OCCUPANCY_THRESHOLD:
                        last_viable_comfort_mode_index = round(index/5)
                        break

                for start in range(0, len(df.loc[day]), 5)[first_viable_comfort_mode_index:last_viable_comfort_mode_index]:
                    for end in range(0, len(df.loc[day]), 5)[round(start/5):last_viable_comfort_mode_index]:
                        occupancy_eco = df.loc[day][:start].occupancy.tolist()
                        occupancy_comfort = df.loc[day][start:end].occupancy.tolist()
                        occupancy_eco.extend(df.loc[day][end:].occupancy.tolist())
                        current_score = self.kpi_calculator.score_scheme(occupancy_eco, occupancy_comfort)
                        if current_score > best_score:
                            best_score = current_score
                            best_comfort_timespan = [start, end]
                            best_occupancy_eco = occupancy_eco
                            best_occupancy_comfort = occupancy_comfort

                total_occupancy_eco.extend(best_occupancy_eco)
                total_occupancy_comfort.extend(best_occupancy_comfort)
                print(day)
                self.kpi_calculator.calculate_average_occupancy(best_occupancy_eco, best_occupancy_comfort)
                print('Score: ', best_score)
                print()
                scheme.extend([0] * best_comfort_timespan[0])
                scheme.extend([1] * (best_comfort_timespan[1] - best_comfort_timespan[0]))
                scheme.extend([0] * (720 - best_comfort_timespan[1]))

        print('Total')
        self.kpi_calculator.calculate_average_occupancy(total_occupancy_eco, total_occupancy_comfort)
        score = self.kpi_calculator.score_scheme(total_occupancy_eco, total_occupancy_comfort)
        print('Score: ', score)
        print()

        return scheme

    def prepare_scheme_for_chart(self, scheme):
        prepared_scheme = []
        for day in self.DAYS:
            previous_timestamp = 0
            for timespan in scheme[day]:
                # Turns timestamp into how many times 2 minutes fit into it so it lines with the measurements
                t1 = datetime.datetime.strptime(timespan['ends_on'], '%H:%M:%S')
                t2 = datetime.datetime(1900,1,1)
                time_delta = (t1 - t2).total_seconds()
                timestamp = int(time_delta // 120 + (time_delta % 120 > 0))

                if timespan['preset']['name'] == 'Eco':
                    prepared_scheme.extend([0] * (timestamp - previous_timestamp))
                else:
                    prepared_scheme.extend([1] * (timestamp - previous_timestamp))
                previous_timestamp = timestamp

        return prepared_scheme
                



    def chart_scheme_and_occupancy(self, scheme, df):
        weekly_occupancy = []
        top_line = [1] * 5040
        bottom_line = [0] * 5040
        format_line = ([1, 0] + [self.OCCUPANCY_THRESHOLD] * 718) * 7
        for day in self.DAYS:
            if day in df.index:
                weekly_occupancy.extend(df.loc[day]['occupancy'].tolist())
            else:
                weekly_occupancy.extend(df.loc[day]['occupancy'].tolist())

        
        plt.plot(format_line, 'red', linewidth = .5)
        plt.plot(weekly_occupancy, 'blue')
        plt.fill_between(range(5040), scheme, bottom_line, color = 'yellow', alpha = .5)
        plt.fill_between(range(5040), scheme, top_line, color = 'green', alpha = .3)
        tick_names = [[day[:3], '6:00', '12:00', '18:00'] for day in self.DAYS]
        tick_names = [item for sublist in tick_names for item in sublist]
        ticks = [x * 180 for x in range(28)]
        plt.xticks(ticks, tick_names)
        plt.yticks([x / 10 for x in range(11)], [str(x * 10) + '%' for x in range(11)])
        plt.ylabel('Bezettingsgraad')
        plt.xlabel('Timestamp')
        plt.grid()
        plt.show()

    def calculate_scores_per_hour(self, df, original_scheme):
        scores_per_hour = []
        for day in [day for day in self.DAYS if day in df.index]:
            for hour in range(24):
                start = hour * 30
                end = (hour + 1) * 30
                # First calculates the score for the scheme while only looking at a single hour
                # Then creates an dictionary entry using the score as key and the day and hour as value
                scores_per_hour.append([
                    self.kpi_calculator.score_scheme(
                        *self.kpi_calculator.calculate_occupancy_per_mode(df.loc[day][start:end], original_scheme[day])),
                    day, hour])
        # List of the scores per hour sorted from lowest to highest
        scores_per_hour.sort(key=lambda x: x[0])
        return scores_per_hour
        
    def suggest_changes(self, scores_per_hour, amount=20):
        print('Scheme hours with the lowest score: ')
        for i in range(amount if amount < len(scores_per_hour) else len(scores_per_hour)):
            score = scores_per_hour[i]
            print(str(i + 1) + '. Score:', round(score[0], 2), 'on', score[1], str(score[2]) + ':00')





