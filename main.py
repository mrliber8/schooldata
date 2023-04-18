from predictor.Co2Predictor import Co2Predictor
from predictor.DataframeManipulator import DataframeManipulator
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from predict_and_show import predict_and_show as pas


def main():


    df_man = DataframeManipulator()
    pred = Co2Predictor()

    train_df = df_man.prepare_dataframe("csv/train.csv")
    validation_df = df_man.prepare_dataframe("csv/validation.csv")

    print(train_df.head())

    predicted_val_y = pred.predict(train_df, validation_df)
    predicted_val_y = 0
    pas().setup_co2(predicted_val_y)


if __name__ == "__main__":
    main()

