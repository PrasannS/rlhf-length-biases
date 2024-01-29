import requests, json

def get_scores_from_api(text_list, url):
    # URL of your Flask API
    # url = 'http://127.0.0.1:5000/predict'

    # Prepare data in the correct format
    data = json.dumps({"texts": text_list})

    # Send POST request to the Flask API
    try:
        response = requests.post(url, data=data, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Get the scores from the response
        scores = response.json()
        return scores
    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}")
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}")
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}")
    except requests.exceptions.RequestException as err:
        print(f"Oops: Something Else: {err}")
        
if __name__=="__main__":
    print(get_scores_from_api(["hi there"]*100, "http://127.0.0.1:5001/predict"))