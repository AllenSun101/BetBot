import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('WEATHER_API_KEY')

def get_weather_statistics(location: str, date: str):
    request_url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{location}/{date}?unitGroup=us&include=current&key={API_KEY}&contentType=json"
    response = requests.request("GET", request_url)
    if response.status_code != 200:
        print('Search Error: ', response.status_code)

    data = response.json()
    output = {
        "latitude": data["latitude"],
        "longitude": data["longitude"],
        "temperature": data["currentConditions"]["temp"],
        "feels_like": data["currentConditions"]["feelslike"],
        "humidity": data["currentConditions"]["humidity"],
        "precip": data["currentConditions"]["precip"],
        "precip_prob": data["currentConditions"]["precipprob"],
        "windgust": data["currentConditions"]["windgust"],
        "windspeed": data["currentConditions"]["windspeed"],
        "visibility": data["currentConditions"]["visibility"],
    }

    return output

if __name__ == "__main__":
    print(get_weather_statistics("Houston Texas", "2026-02-10T05:00:00"))