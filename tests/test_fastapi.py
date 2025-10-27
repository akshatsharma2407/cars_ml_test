import unittest
from fastapi.testclient import TestClient
from fastapi_app.app import app 

class FastAPITests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_predict_endpoint(self):
        response = self.client.post("/predict", json={
                                                    "Model_Year": 2020,
                                                    "Mileage": 18000,
                                                    "Accidents_Or_Damage": 0,
                                                    "Clean_Title": 1,
                                                    "One_Owner_Vehicle": 1,
                                                    "Personal_Use_Only": 1,
                                                    "Level2_Charging": 0,
                                                    "Dc_Fast_Charging": 1,
                                                    "Battery_Capacity": 75,
                                                    "Expected_Range": 350,
                                                    "Gear_Spec": 1,
                                                    "Engine_Size": 2.0,
                                                    "Valves": 16
                                                    })
        
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()
