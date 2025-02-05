import os
import sys
from api.app import app
from scripts.generate_outputs import generate_descriptions

def main():
    if len(sys.argv) > 1:
        # If 'generate' command is passed, generate descriptions for services
        if sys.argv[1] == "generate":
            generate_descriptions()
        else:
            print("Invalid argument. Use 'generate' to generate service descriptions.")
    else:
        # Otherwise, run the Flask app on port 5001
        print("Starting Flask app on port 5001...")
        app.run(debug=True, port=5001)  # Set port to 5001

if __name__ == '__main__':
    main()