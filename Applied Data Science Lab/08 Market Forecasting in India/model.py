import os
from glob import glob
from datetime import datetime

import joblib
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult

from config import settings


class GarchModel:

    def __init__(self, ticker, repo, use_new_data=False):
        """
        Initialize GARCH model.

        Parameters
        ----------
        ticker : str
            Stock ticker symbol.
        repo : SQLRepository
            Database repository object.
        use_new_data : bool
            Whether to fetch new API data.
        """

        self.ticker = ticker
        self.repo = repo
        self.use_new_data = use_new_data
        self.model_directory = settings.model_directory


    def wrangle_data(self, n_observations=1000):
        """
        Get and prepare stock return data.
        """

        if self.use_new_data:

            df = self.repo.get_daily_data(
                ticker=self.ticker,
                output_size="full"
            )

        else:

            df = self.repo.read_table(
                table_name=self.ticker,
                limit=n_observations + 1
            )


        # Calculate returns

        df = df.sort_index()

        df["return"] = (
            df["close"]
            .pct_change()
            .mul(100)
        )


        # Remove missing values

        self.data = (
            df["return"]
            .dropna()
            .tail(n_observations)
        )

        return self.data



    def fit(self, p=1, q=1):
        """
        Fit GARCH model.
        """

        if not hasattr(self, "data"):
            raise Exception(
                "Run wrangle_data before fitting model."
            )


        model = arch_model(
            self.data,
            p=p,
            q=q,
            rescale=False
        )


        self.model = model.fit(
            disp=0
        )


        return self.model



    def predict_volatility(self, horizon=5):
        """
        Forecast future volatility.
        """

        if not hasattr(self, "model"):
            raise Exception(
                "Fit model before prediction."
            )


        forecast = self.model.forecast(
            horizon=horizon
        )


        variance = (
            forecast
            .variance
            .iloc[-1]
        )


        prediction = {}

        for date, value in variance.items():

            prediction[str(date)] = float(
                value ** 0.5
            )


        return prediction



    def dump(self):
        """
        Save trained model using joblib.
        """

        if not hasattr(self, "model"):
            raise Exception(
                "No model to save."
            )


        os.makedirs(
            self.model_directory,
            exist_ok=True
        )


        timestamp = datetime.now().isoformat()

        filename = os.path.join(
            self.model_directory,
            f"{timestamp}_{self.ticker}.pkl"
        )


        joblib.dump(
            self.model,
            filename
        )


        return filename



    def load(self):
        """
        Load latest saved model.
        """


        pattern = os.path.join(
            self.model_directory,
            f"*{self.ticker}.pkl"
        )


        try:

            model_path = sorted(
                glob(pattern)
            )[-1]


        except IndexError:

            raise Exception(
                f"No model trained for '{self.ticker}'."
            )


        self.model = joblib.load(
            model_path
        )


        return self.model