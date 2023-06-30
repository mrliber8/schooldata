import datetime
import requests


class KPICalculator:
    def __init__(self, occupancy_threshold) -> None:
        # The percentage of occupancy when you want to use cofort mode rather than eco.
        # In other words, setting this to .5 means that kpi calculator gives more score to eco mode
        #  when the occupancy is lower than 50% and more score to comfort mode when the occupancy is higher than 50%
        self.OCCUPANCY_THRESHOLD = occupancy_threshold
        pass

    def calculate_kpi(self, df):
        """
        Calculates the occupancy rate for both the eco and comfort mode.

        Firstly the occupancy per week day is calculated. Here we take every measurement of the same weekday and same timestamp
         within the given dataframe and take the avarage of each measurements occupancy. For example, if we have a dataframe
         with the measurements of a period of 4 weeks, each weekday will occur 4 times. Then we take all the mondays
         and group then together. If in those 4 mondays there are 3 days where at 12:00 there is registered occupancy
         and 1 monday where at 12:00 there isn't any occupancy then 12:00 on monday will be given an occupancy of 0.75.
        This will be done for all timestamp for each day.

        These occupancy values will then be grouped into either the eco or the comfort mode depending on what mode is active
         according to the scheme which controls temperature. An average of these values will then be calculated to get the
         occupancy rate for both modes.
        In a perfect situation you want the occupancy rate for eco to be as close to 0% as possible
         and for comfort mode to be as close to 100%.

        This will also give a score from 0 to 1 indicating how well the scheme overlap with the measurements. This will use
         the same occupancy values as before. Here we take an average of all the occupanct values. But before we do that we
         subtract the occupancy values of all the measurement which fall within eco mode according to scheme from 1. This way,
         if there is an occupancy value of 0 during eco mode, it will be turned into a 1 before taking an average.
        """
        self.calculate_occupancy(df)
        df = self.group_by_weekdays(df)
        scheme = self.get_scheme()
        self.compare_scheme_and_historical_data(df, scheme)
        return df

    def get_scheme(self):
        """
        Requests the scheme controlling the heaters from the climatics api and returns
        """
        r = requests.get('https://climatics.nl/api/location/11/schedule/87', headers={'Authorization': '3b44a419-95ac-4ae3-b9e9-00774a992eee'})
        return r.json()

    def calculate_occupancy(self, df):
        """
        Derives the occupancy from the co2 values
        """
        # occupancy_predictor = Prediction()
        # df = occupancy_predictor.main(df)
        # df = df.rename(columns={'in_room': 'occupancy'})

        # Set the window Size and the CO2 treshold
        window_size = 30
        co2_threshold = 500

        # Calculate the rolling mean and median
        df['rolling_mean'] = df['co2'].rolling(window=window_size, min_periods=1).mean()
        df['rolling_median'] = df['co2'].rolling(window=window_size, min_periods=1).median()

        # Condition to determine if someone is in the room
        condition = (df['co2'] > co2_threshold) & (df['rolling_mean'] > co2_threshold) & (
                    df['rolling_median'] > co2_threshold)

        # Set the value as an int
        df['occupancy'] = condition.astype(int)

        #df['occupancy'] = [int(x > 500) for x in df[df.columns[0]]]
        df = df.drop('rolling_mean', axis=1)
        df = df.drop('rolling_median', axis=1)
        del df[df.columns[0]]

    def group_by_weekdays(self, df):
        """
        Modifies the dataframe by grouping the records based on the weekday of the measurement
        Every measurement on the same time and the same weekday but on a different week,
         will have an average taken of it's occupancy

        This means that if there is a friday where at 12:00 a room is occupied
         and that same room is unoccupied the next friday at 12:00,
         then there will be an average occupancy of 0.5
        """
        # We use the name of the weekday instead of a number 1 to 7
        #  because it makes it easier to use in the compare_scheme_and_historical_data method
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        df["weekday"] = [days[i.weekday()] for i in df.index]

        # Remove the date portion of the datetime so only the time remains
        df.index = df.index.time
        df.index.name = "time"

        df = df.groupby(["weekday", "time"]).mean()
        return df

    def compare_scheme_and_historical_data(self, df, scheme):
        """
        Prints the occupancy percentage during both the eco and comfort mode indicated by the scheme
        """
        occupancy_eco, occupancy_comfort = {}, {}
        # We specify the days of a week because the api returns the days of the week out of order
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        scheme = scheme['days']

        for day in days:
            # These 2 make sure that when grouping the measurements within a certain mode's timespan
            #  the loop start at start of that timespan
            # With modes we mean eco and comfort
            index_last_used_measurement, counter = 0, 0
            occupancy_eco[day], occupancy_comfort[day] = [], []
            for timespan in scheme[day]:
                occupancy = []
                # This for loop groups all the measurements within a certain mode's timespan
                for measurement in df.loc[day][index_last_used_measurement:].iloc:
                    # The if is to determine whether measurments were made before the end of the timespan of a certain mode
                    if(measurement.name < datetime.datetime.strptime(timespan['ends_on'], '%H:%M:%S').time()):
                        occupancy.append(measurement.occupancy)
                        counter += 1
                    else:
                        index_last_used_measurement = counter
                        break

                # Adds the grouped measurments to either the eco or the comfort list
                #  depending on which mode was active at according to the scheme
                if timespan['preset']['name'] == 'Eco':
                    occupancy_eco[day].extend(occupancy)
                else:
                    occupancy_comfort[day].extend(occupancy)

        # Prints the occupancy percentages
        total_occupancy_comfort = []
        for day in occupancy_comfort.values():
            total_occupancy_comfort.extend(day)

        total_occupancy_eco = []
        for day in occupancy_eco.values():
            total_occupancy_eco.extend(day)

        print('Total')
        print('comfort: ', round(sum(total_occupancy_comfort) / len(total_occupancy_comfort) * 100), '% bezetting')
        print('eco:     ', round(sum(total_occupancy_eco) / len(total_occupancy_eco) * 100), '% bezetting')
        print()
        score = 0
        for occupancy in total_occupancy_comfort:
            score -= self.calculated_score_by_measurement(occupancy)
        for occupancy in total_occupancy_eco:
            score += self.calculated_score_by_measurement(occupancy)
        score += len(total_occupancy_comfort)
        score /= len(total_occupancy_comfort) + len(total_occupancy_eco)
        print('Score: ', score)

        print('Bezettings graad per mode in %:')
        print('day - comfort - eco')
        for day in days:
            print(day, ' ' * (9 - len(day)),
                  round(sum(occupancy_comfort[day]) / len(occupancy_comfort[day]) * 100)
                        if len(occupancy_comfort[day]) != 0 else '100',
                  round(sum(occupancy_eco[day]) / len(occupancy_eco[day]) * 100)
                        if len(occupancy_eco[day]) != 0 else '100',
            )

    def score_generated_scheme(self, df, start, end):
        score = 0
        for occupancy in df[:start].occupancy:
            score -= self.calculated_score_by_measurement(occupancy)
        for occupancy in df[start:end].occupancy:
            score += self.calculated_score_by_measurement(occupancy)
        for occupancy in df[end:].occupancy:
            score -= self.calculated_score_by_measurement(occupancy)
        score += len(df[:start]) + len(df[end:])
        score /= len(df)
        # print (score)
        return score 

    def calculated_score_by_measurement(self, occupancy):
        if occupancy < self.OCCUPANCY_THRESHOLD:
            return occupancy / self.OCCUPANCY_THRESHOLD / 2
        else:
            return (occupancy - self.OCCUPANCY_THRESHOLD) / (1 - self.OCCUPANCY_THRESHOLD) / 2 + .5
        




