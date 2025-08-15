import re
import sys

def extract_token_ids(file_path):
    """
    Reads a log file and extracts all 'Generated token ID' values.

    Args:
        file_path (str): The path to the input log file.

    Returns:
        list: A list of integers representing the extracted token IDs.
    """
    # This regular expression looks for the line "Generated token ID: "
    # and captures the sequence of digits that follows.
    token_pattern = re.compile(r"Generated token ID: (\d+)")
    
    token_ids = []
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = token_pattern.search(line)
                if match:
                    # If a match is found, convert the captured group (the digits)
                    # to an integer and add it to our list.
                    token_id = int(match.group(1))
                    token_ids.append(token_id)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
        
    return token_ids

if __name__ == "__main__":
    # Check if a file path was provided as a command-line argument.
    if len(sys.argv) < 2:
        print("Usage: python extract_script.py <path_to_log_file>")
    else:
        log_file = sys.argv[1]
        tokens = extract_token_ids(log_file)
        
        if tokens is not None:
            print("Extracted Token IDs:")
            print(tokens)
