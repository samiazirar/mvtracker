import os
from dotenv import load_dotenv
from huggingface_hub import login

# Load the .env file
load_dotenv()

# Get the Hugging Face token from the .env file
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    # Authenticate with Hugging Face
    login(token=hf_token)
    print("Successfully authenticated with Hugging Face!")
else:
    print("Hugging Face token not found in .env file.")