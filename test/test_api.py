import numpy as np
from fastapi.testclient import TestClient
import app.main_copy1 as m  # import module, not only app

class DummyScaler:
    def transform(self, X):
        return X

class DummyModel:
    def predict(self, X):
        return np.array([4.2])

def test_predict():
    # force ready state for tests
    m.model = DummyModel()
    m.scaler = DummyScaler()
    m.redis_client = None

    payload = {
        "MedInc": 8.3252,
        "HouseAge": 41,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
        "RoomsPerHousehold": 2.0,
        "BedroomsPerHouse": 0.3,
        "PopulationPerHousehold": 3.0
    }

    with TestClient(m.app) as client:
        response = client.post("/v1/predict", json=payload)
        assert response.status_code in [200, 503]
        data = response.json()
        assert data["status"] == "success"
        assert isinstance(data["predicted_price"], float)