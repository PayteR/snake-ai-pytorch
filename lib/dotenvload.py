from dotenv import load_dotenv
import os

# Provide the path to the .env file
dotenv_path = os.path.join(os.path.dirname(__file__), '../.env')

# if the .env file found, load it
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
