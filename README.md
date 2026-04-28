# Financial Advisor Chatbot

## Overview

The Financial Advisor Chatbot is a project designed to provide users with quick and accurate answers to their questions related to public companies listed on NASDAQ. The chatbot leverages advanced technologies in finance and AI, including LangChain, FastAPI, and ChromaDB, to deliver a seamless user experience.

## Project Structure

The project is organized into several components:

- **api**: Contains the FastAPI application, including API endpoints, services, database models, and utility functions.
- **front-chat**: Contains the user interface for interacting with the chatbot.
- **backend-api**: Contains the backend of the application.
- **populate_chroma**: Contains the required configuration and Docker setup scripts to populate the ChromaDB database.
- **preprocess**: Contains the configuration and required script to preprocess PDF inputs into Markdown files.

## Additional files and directories

- **EDA.ipynb**: This file contains an Exploratory Data Analysis (EDA). This notebook is used to analyze and visualize the dataset, providing insights and understanding of the data before it is processed and used by the application. It includes various data analysis techniques and visualizations to help identify patterns, trends, and anomalies in the data.

- **evaluation**: This directory contains a notebook called `Evaluations.ipynb` to perform model evaluation of the LLM application within this project. Requires the initialization of a virtual environment and running `pip install -r requirements.txt` within this directory to successfully run the notebook.

## Setup Instructions

### Prerequisites

- Docker and Docker Compose installed on your machine.

## To set up the application and populate the ChromaDB vector database

### Steps

1. Clone the repository in a local folder of your own choosing:

   ```git
   git clone <repository-url>
   cd financial-advisor-chatbot
   ```

2. Create the `.env` file from `.env.original`. Standing on the project's root, run

    ```shell
    cp .env.original .env
    ```

3. Set the value of the `OPENAI_API_KEY` variable inside your newly created `.env` file.

4. Repeat steps 2 and 3 for the `.env.original` files found in the `front-chat` and `backend-api` directories, respectively.

5. You need to place the `dataset/` directory within the `preprocess/` folder. The `dataset/` directory must contain all of the PDFs you wish to use for population of the Chroma vector database.

6. Run the `preprocess.sh` bash script. This script will convert all newly added PDFs to the `dataset/` directory to markdown format and store them inside a pre-defined docker volume that the overall app already has access to.

7. When the previous script is done running, run the app with

    ```shell
     docker-compose up --build -d populate_chroma
    ```

   This will populate the vector database using the generated markdown files inside the `populate_chroma` container.

8. Once the `populate_chroma` container finishes execution (it can take quite a while, depending on the amount of new PDFs in the `dataset/` directory), the application is ready to go, as the vector database is now populated with the new document embeddings.

9. Stop the current `docker-compose` execution by running

    ```shell
    docker-compose down
    ```

10. You can now run the application with

    ```shell
    docker-compose up --build -d
    ```

### Running the Application (after populating the ChromaDB vector database)

1. Standing on the root of the project, build and run the services using Docker Compose:

   ```shell
   docker-compose up --build -d
   ```

2. Access the frontend application at `http://localhost:8501`.

3. The FastAPI backend can be accessed at `http://localhost:8002/docs` for API documentation.

## Usage

- Users can interact with the chatbot through the frontend interface, asking questions related to NASDAQ companies.
- The backend processes these queries, retrieves relevant information from the vector database, and generates responses using LangChain.

## Note: To get the content of ChromaDB to persist across re-deploys

You have to create the `PERSIST_DIRECTORY` environment variable in your `.env` file and set its value to `/chroma/whatever-you-want` (the default path where ChromaDB stores its data is `./chroma`, as stated [here](https://cookbook.chromadb.dev/core/storage-layout/)). That value is going to be the path inside the ChromaDB container where our named volume is going to be mounted. For example, `/chroma/my_db`. Note: you **cannot** set this variable as simply `/chroma/` or any such variation, as Docker will throw an error upon running the containers.

Then, in your `docker-compose.yml` file, set the ChromaDB service as follows:

```docker
chromadb:
    container_name: chromadb
    image: chromadb/chroma:latest
    volumes:
      - index_data:/chroma/.chroma/index
      - chroma_persist_storage:${PERSIST_DIRECTORY}
    ports:
      - "8000:8000"
    env_file:
      - ./.env
```

This will automatically create a named volume called `your-app-name_chroma_persist_storage` on your first `up`, and that's where your embeddings are going to live.
