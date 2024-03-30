from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_healthcheck():
    response = client.get("/healthcheck/")
    assert response.status_code == 200
    assert response.json() == {"message": "Service is ready"}
