# AutoML Project

## Overview

This project is a comprehensive Automated Machine Learning (AutoML) platform designed to streamline the machine learning workflow from data preparation to model deployment. It integrates various functionalities including automated data cleaning, supervised and unsupervised learning model training, an AI-powered SQL assistant, and an interactive web-based frontend for user interaction and visualization.

## Features

*   **Automated Data Cleaning:** Utilities to preprocess and clean raw datasets, ensuring data quality for model training.
*   **Supervised Learning Models:** Implementation and integration of various supervised machine learning algorithms.
*   **Unsupervised Learning Models:** Support for unsupervised learning techniques for tasks like clustering and dimensionality reduction.
*   **AI SQL Assistant (Agentic Capability):** A Retrieval Augmented Generation (RAG) based AI assistant to help with SQL queries and database interactions. This component demonstrates agentic capabilities by intelligently processing natural language queries, retrieving relevant information, and generating actionable SQL.
*   **Interactive Web Frontend:** A user-friendly web interface built with HTML, CSS, and JavaScript for interacting with the AutoML functionalities and visualizing results.
*   **Data Visualization:** Tools to generate insightful charts and graphs from processed data and model outputs.

## Project Structure

The project is organized into the following main directories:

*   `.env`: Environment variables, including API keys.
*   `app.py`: The main application entry point.

*   `config.py`: Configuration settings for the application.
*   `frontend/`: Contains the static files for the web-based user interface (HTML, CSS, JavaScript, images).
*   `models/`: Houses the implementations for supervised and unsupervised machine learning models.
*   `rag/`: Contains modules related to the Retrieval Augmented Generation (RAG) system, including memory management and query processing.
*   `utils/`: Utility functions for data cleaning, metrics calculation, and other common tasks.
*   `visuals/`: Modules dedicated to data visualization and chart generation.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Al1Abdullah/AutoML.git
    cd AutoML
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    Create or update the `.env` file in the root directory with your Groq API key:
    ```
    GROQ_API_KEY="YOUR_GROQ_API_KEY_HERE"
    ```
    Similarly, update `groq_config.json` with your Groq API key:
    ```json
    {
      "GROQ_API_KEY": "YOUR_GROQ_API_KEY_HERE"
    }
    ```
    **Note:** Replace `"YOUR_GROQ_API_KEY_HERE"` with your actual Groq API key. Do not commit your actual API keys to version control.

## Usage

To run the AutoML application:

1.  **Activate your virtual environment** (if not already active).
2.  **Run the main application file:**
    ```bash
    python app.py
    ```
    (Further instructions on how to access the web frontend would depend on how `app.py` serves it. If it's a Flask/Django app, it would typically mention a local server address.)

## Technologies Used

*   **Python:** Core programming language.
*   **HTML, CSS, JavaScript:** For the frontend development.
*   **Git:** Version control.
*   **Groq API:** For AI-powered functionalities (e.g., SQL assistant).
*   **CatBoost:** (Implied by `catboost_info`) A machine learning library.

## Future Enhancements (Autonomous System Potential)

The architecture of this project, particularly the RAG-based AI SQL Assistant, lays the groundwork for developing more autonomous capabilities. Future enhancements could involve integrating more complex decision-making processes, self-correction mechanisms, and broader task automation, moving towards a more fully autonomous AutoML system.

## Contributing

Contributions are welcome! Please feel free to fork the repository, create a new branch, and submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details (if applicable).
