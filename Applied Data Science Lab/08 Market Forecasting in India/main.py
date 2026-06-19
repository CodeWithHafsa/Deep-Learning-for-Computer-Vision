from fastapi import FastAPI
from pydantic import BaseModel
import sqlite3

from config import settings
from data import SQLRepository
from model import GarchModel

# =====================================================
# FastAPI App
# =====================================================

app = FastAPI()


# =====================================================
# Input / Output Schemas
# =====================================================

class FitIn(BaseModel):
    ticker: str
    use_new_data: bool
    n_observations: int
    p: int
    q: int


class FitOut(FitIn):
    success: bool
    message: str


class PredictIn(BaseModel):
    ticker: str
    n_days: int


class PredictOut(PredictIn):
    success: bool
    forecast: dict
    message: str


# =====================================================
# Helper Function
# =====================================================

def build_model(ticker: str, use_new_data: bool) -> GarchModel:
    """
    Create GarchModel instance with SQL repository.
    """

    connection = sqlite3.connect(
        settings.db_name,
        check_same_thread=False
    )

    repo = SQLRepository(connection=connection)

    model = GarchModel(
        ticker=ticker,
        repo=repo,
        use_new_data=use_new_data
    )

    return model


# =====================================================
# Routes
# =====================================================

@app.get("/hello")
def hello():
    return {"message": "Hello world!"}


@app.post("/fit", response_model=FitOut)
def fit_model(request: FitIn):

    try:
        model = build_model(
            ticker=request.ticker,
            use_new_data=request.use_new_data
        )

        model.wrangle_data(
            n_observations=request.n_observations
        )

        model.fit(
            p=request.p,
            q=request.q
        )

        filename = model.dump()

        return FitOut(
            ticker=request.ticker,
            use_new_data=request.use_new_data,
            n_observations=request.n_observations,
            p=request.p,
            q=request.q,
            success=True,
            message=f"Trained and saved '{filename}'."
        )

    except Exception as e:

        return FitOut(
            ticker=request.ticker,
            use_new_data=request.use_new_data,
            n_observations=request.n_observations,
            p=request.p,
            q=request.q,
            success=False,
            message=str(e)
        )


@app.post("/predict", response_model=PredictOut)
def predict(request: PredictIn):

    try:
        model = build_model(
            ticker=request.ticker,
            use_new_data=False
        )

        model.load()

        forecast = model.predict_volatility(
            horizon=request.n_days
        )

        return PredictOut(
            ticker=request.ticker,
            n_days=request.n_days,
            success=True,
            forecast=forecast,
            message=""
        )

    except Exception as e:

        return PredictOut(
            ticker=request.ticker,
            n_days=request.n_days,
            success=False,
            forecast={},
            message=str(e)
        )