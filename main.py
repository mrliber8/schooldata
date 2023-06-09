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
    t_start = int(time.mktime(datetime.datetime(2022, 1, 1).timetuple()))
    t_end = int(time.mktime(datetime.datetime(2023, 1, 1).timetuple()))

    # Validation dataframe
    v_start = int(time.mktime(datetime.datetime(2023, 1, 1).timetuple()))
    v_end = int(time.mktime(datetime.datetime(2023, 2, 1).timetuple()))

    # The id of the sensor on which a prediction should be made
    sensor_id = 3137

    # Whether you want to predict co2 values, calculate kpi or both
    x = [False, True]


    retriever = Retriever(api_key="3b44a419-95ac-4ae3-b9e9-00774a992eee", location_id=11)
    train = retriever.get_logs_for_sensor_list(sensor_ids=[sensor_id], start=t_start, end=t_end, as_pandas=True, interpolate=True)
    validation = retriever.get_logs_for_sensor_list(sensor_ids=[sensor_id], start=v_start, end=v_end, as_pandas=True, interpolate=True)

    if x[0]:
        print("Prediction results:")
        df_man = DataframeManipulator()
        pred = Co2Predictor()

        train_df = df_man.prepare_dataframe(train)
        validation_df = df_man.prepare_dataframe(validation)

        pred.predict(train_df, validation_df)
        print()

    if x[1]:
        print("Kpi results:")
        kpi_calculator = KPICalculator()
        df = kpi_calculator.calculate_kpi(train)
        scheme_generator = SchemeGenerator()
        scheme_generator.generate_scheme(df)
        print()

if __name__ == "__main__":
    main()

