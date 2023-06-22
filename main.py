from kpi_calculator.KPICalculator import KPICalculator
from kpi_calculator.SchemeGenerator import SchemeGenerator
from predictor.Co2Predictor import Co2Predictor
from predictor.DataframeManipulator import DataframeManipulator
import datetime
import time
from climatics_client.retrieve import Retriever


def main():
    start = time.time()
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

    # The percentage of occupancy when you want to use cofort mode rather than eco.
    # In other words, setting this to .5 means that kpi calculator gives more score to eco mode
    #  when the occupancy is lower than 50% and more score to comfort mode when the occupancy is higher than 50%
    OCCUPANCY_THRESHOLD = 0.5


    retriever = Retriever(api_key="3b44a419-95ac-4ae3-b9e9-00774a992eee", location_id=11)
    train = retriever.get_logs_for_sensor_list(sensor_ids=[sensor_id], start=t_start, end=t_end, as_pandas=True, interpolate=True)
    train.rename(columns={train.columns[0]: 'co2'}, inplace=True)
    validation = retriever.get_logs_for_sensor_list(sensor_ids=[sensor_id], start=v_start, end=v_end, as_pandas=True, interpolate=True)
    validation.rename(columns={validation.columns[0]: 'co2'}, inplace=True)

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
        kpi_calculator = KPICalculator(OCCUPANCY_THRESHOLD)
        df = kpi_calculator.calculate_kpi(train)
        scheme_generator = SchemeGenerator(OCCUPANCY_THRESHOLD)
        scheme_generator.generate_scheme(df)
        print()

    end = time.time()
    print("Elapsed time in seconds of the main function is: ", end - start)

if __name__ == "__main__":
    main()

