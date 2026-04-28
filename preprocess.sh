#!/bin/bash

# Load environment variables from .env file
export $(grep -v '^#' ./.env | xargs)

# Set default values for environment variables
DATASET_DIRECTORY=${DATASET_DIRECTORY:-dataset}
DATASET_MD_DIRECTORY=${DATASET_MD_DIRECTORY:-dataset_md}

# cd to the preprocess directory
cd preprocess
docker build -f ./Dockerfile.preprocess -t preprocess .

# Run preprocess container
cd ..
docker run --name preprocess_container -d --env-file ./.env -v source_data:/preprocess/${DATASET_DIRECTORY} -v preprocess_destination_data:/preprocess/${DATASET_MD_DIRECTORY} preprocess

docker wait preprocess_container
