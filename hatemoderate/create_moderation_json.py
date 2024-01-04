from openai import OpenAI
import pandas as pd
import json

client = OpenAI(api_key="YOUR_API_KEY")

# Function to create a JSON file with the moderation results returned by the API
def process_example(row):
    response = client.moderations.create(input=row['sentence'])
    moderation_result = {
        "id": row['example_id'],
        "categories": response.results[0].categories.__dict__,
        "category_scores": response.results[0].category_scores.__dict__,
        "flagged": response.results[0].flagged
    }
    return moderation_result

if __name__ == "__main__":
    # Read csv file of all examples
    examples = pd.read_csv("data/all_examples.csv", delimiter='\t')

    # Apply the function created to all the examples
    moderation_results = examples.apply(process_example, axis=1)

    json_data = json.dumps(moderation_results.tolist(), indent=4)

    # Create a JSON file and write all the data
    with open('moderation_results.json', 'w') as json_file:
        json_file.write(json_data)
        
    print("JSON file created successfully.")