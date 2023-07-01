from kpi_calculator.KPICalculator import KPICalculator
from kpi_calculator.SchemeGenerator import SchemeGenerator
from predictor.Co2Predictor import Co2Predictor
from predictor.DataframeManipulator import DataframeManipulator
import csv
import matplotlib.pyplot as plt
import datetime
import time
import numpy as np
from sklearn.linear_model import LinearRegression
from predict_and_show import predict_and_show as pas
from climatics_client.retrieve import Retriever
import pprint
import arrow
from climatics_client.utils import get_timestamps_of_date
from var_dump import var_dump
import requests


def main():
    # Train dataframe
    T_START = int(time.mktime(datetime.datetime(2022, 1, 1).timetuple()))
    T_END = int(time.mktime(datetime.datetime(2023, 1, 1).timetuple()))

    # Validation dataframe
    V_START = int(time.mktime(datetime.datetime(2023, 1, 1).timetuple()))
    V_END = int(time.mktime(datetime.datetime(2023, 2, 1).timetuple()))

    # The id of the building which you wanna use
    LOCATION_ID = 11

    # The id of the sensor which you wanna use
    SENSOR_ID = 3137

    # The id of the scheme which you wanna use
    SCHEME_ID = 87

    # Whether you want to predict co2 values, calculate kpi or both
    x = [False, True]

    # The percentage of occupancy when you want to use cofort mode rather than eco.
    # In other words, setting this to .5 means that kpi calculator prefers eco mode
    #  when the occupancy is lower than 50% and prefers comfort mode when the occupancy is higher than 50%
    OCCUPANCY_THRESHOLD = 0.5

    # Get train and validation dataframes from the climatics client API
    retriever = Retriever(api_key="3b44a419-95ac-4ae3-b9e9-00774a992eee", location_id=LOCATION_ID)
    train = retriever.get_logs_for_sensor_list(sensor_ids=[SENSOR_ID], start=T_START, end=T_END, as_pandas=True, interpolate=True)
    train.rename(columns={train.columns[0]: 'co2'}, inplace=True)
    validation = retriever.get_logs_for_sensor_list(sensor_ids=[SENSOR_ID], start=V_START, end=V_END, as_pandas=True, interpolate=True)
    validation.rename(columns={validation.columns[0]: 'co2'}, inplace=True)


    if x[0]:
        print("Prediction results:")
        df_man = DataframeManipulator()
        pred = Co2Predictor()

        train_df = df_man.prepare_dataframe(train)
        validation_df = df_man.prepare_dataframe(validation)

        print(pred.predict(train_df, validation_df))
        print()

    if x[1]:
        print("Kpi results:")
        kpi_calculator = KPICalculator(OCCUPANCY_THRESHOLD, LOCATION_ID, SCHEME_ID)
        df = kpi_calculator.calculate_kpi(train)

        scheme_generator = SchemeGenerator(OCCUPANCY_THRESHOLD, kpi_calculator)

        original_scheme = kpi_calculator.get_scheme()
        scores_per_hour = scheme_generator.calculate_scores_per_hour(df, original_scheme)
        scheme_generator.suggest_changes(scores_per_hour)
        prepared_original_scheme = scheme_generator.prepare_scheme_for_chart(original_scheme)
        
        scheme_generator.chart_scheme_and_occupancy(prepared_original_scheme, df)
        scheme_generator.generate_scheme(df)

        print()

if __name__ == "__main__":
    main()

