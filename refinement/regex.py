import json
import re

def extract_json_with_regex(response: str):
    pattern = r"<output>(.*?)</output>"
    # Search for the pattern <output>...</output>
    match = re.search(pattern, response, re.DOTALL)

    if match:
        # Extract the content between the tags
        json_str = match.group(1).strip()
        try:
            # Parse the string to a JSON object
            json_data = json.loads(json_str)
            return json_data
        except json.JSONDecodeError:
            # Return None if JSON parsing fails
            return None
    # Return None if no match is found
    return None