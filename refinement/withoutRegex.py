import json

def extract_json_without_regex(response: str):
    start_tag = "<output>"
    end_tag = "</output>"
    # Find the start and end indices of the tags
    start_index = response.find(start_tag)
    end_index = response.find(end_tag)

    if start_index != -1 and end_index != -1:
        # Adjust start index to get the content after the start tag
        start_index += len(start_tag)
        # Extract the content between the tags
        json_str = response[start_index:end_index].strip()
        try:
            # Parse the string to a JSON object
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            # Return None if JSON parsing fails
            return None
    # Return None if tags are not found
    return None