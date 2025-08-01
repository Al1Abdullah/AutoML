"""Module for querying the Groq API with dataset context."""

from groq import Groq, APIStatusError
from config import GROQ_API_KEY
from rag.memory import get_dataset
import pandas as pd
import logging

# Configure logging for this module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Groq client with API key from config
client = Groq(api_key=GROQ_API_KEY)

def query_dataset_with_groq(dataset_name, user_query):
    """Queries the Groq API with a user question, providing dataset context.

    Args:
        dataset_name (str): The name of the dataset to retrieve from memory.
        user_query (str): The user's question about the dataset.

    Returns:
        str: The AI's answer to the question, or an error message if the query fails.
    """
    logging.info(f"Attempting to query Groq with user question: {user_query}")
    df = get_dataset(dataset_name)
    if df is None:
        logging.error(f"Dataset '{dataset_name}' not found in memory for Groq query.")
        return "No dataset found with that name. Please upload a dataset first."

    # Prepare context for the LLM, including dataset overview, summary statistics, and a sample
    context = f"""
You are an expert Data Analyst. You have been provided with a dataset.

**Dataset Overview:**
- **Shape:** {df.shape[0]} rows and {df.shape[1]} columns.
- **Columns and Data Types:**\n{df.dtypes.to_string()}

**Summary Statistics:**\n{df.describe(include='all').to_string()}

**First 5 Rows:**\n{df.head(5).to_string(index=False)}

**User Question:** {user_query}

Answer the user's question clearly and accurately based *only* on the provided dataset information.
"""

    try:
        logging.info("Sending request to Groq API for chat completion.")
        response = client.chat.completions.create(
            model="llama3-70b-8192", # Using a powerful model for better understanding
            messages=[
                {"role": "system", "content": "You are a helpful data science assistant. Provide concise and accurate answers."},
                {"role": "user", "content": context}
            ],
            temperature=0.1, # Low temperature for factual and less creative responses
            max_tokens=1024, # Limit response length
            top_p=1,
            stop=None,
        )
        ai_response_content = response.choices[0].message.content
        logging.info("Successfully received response from Groq API.")
        return ai_response_content
    except APIStatusError as e:
        logging.error(f"Groq API error occurred: Status Code {e.status_code}, Response: {e.response}", exc_info=True)
        if e.status_code == 503:
            return "The AI service is currently unavailable due to high demand or maintenance. Please try again later."
        else:
            return f"An error occurred with the AI service (Status: {e.status_code}). Please check the logs for more details."
    except Exception as e:
        logging.error(f"An unexpected error occurred while querying the AI: {e}", exc_info=True)
        return f"An unexpected error occurred while processing your request: {e}"