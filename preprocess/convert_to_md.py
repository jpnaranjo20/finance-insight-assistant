import os
import time
from docling.document_converter import DocumentConverter

# Load dataset directories from environment variables.
# DATASET_DIRECTORY: The directory containing source PDF files.
# DATASET_MD_DIRECTORY: The directory where converted Markdown files will be saved.
DATASET_DIRECTORY = os.getenv("DATASET_DIRECTORY")
DATASET_MD_DIRECTORY = os.getenv("DATASET_MD_DIRECTORY")

# Create an instance of the DocumentConverter to handle PDF-to-Markdown conversion.
converter = DocumentConverter()

# Define the source and destination paths based on the environment variables.
source = f"/preprocess/{DATASET_DIRECTORY}"
destination = f"/preprocess/{DATASET_MD_DIRECTORY}"

start = time.time()

# Iterate over every file in the source directory.
# Each file is expected to be a PDF.
for pdf_file in os.listdir(source):
    try:
        # Build the full path for the current PDF file.
        pdf_path = os.path.join(source, pdf_file)
        # Derive the Markdown filename (replace .pdf with .md).
        md_filename = f"{pdf_file[:-4]}.md"
        
        # Check if the Markdown file already exists in the destination directory.
        if md_filename in os.listdir(destination):
            print(f"Skipping {pdf_file} as it is already processed.")
            continue
        else:
            # Convert the PDF to Markdown using the DocumentConverter.
            result = converter.convert(pdf_path)
            # Save the resulting Markdown document into the destination directory.
            with open(f"{destination}/{md_filename}", "w", encoding="utf-8") as file:
                file.write(result.document.export_to_markdown())
            print(f"Processed {pdf_file}")
    except BaseException as e:
        # If an error occurs during processing, print the error message.
        print(f"Error processing {pdf_file}: {e}")
        # Build the path to the Markdown file that might have been partially created.
        md_file_path = os.path.join(destination, f"{pdf_file[:-4]}.md")
        # If the file exists, remove it to avoid incomplete files.
        if os.path.exists(md_file_path):
            os.remove(md_file_path)
            print(f"Deleted {md_file_path}")
        else:
            print(f"File does not exist: {md_file_path}")
        # Continue with the next file.
        pass

end = time.time()

# Calculate the total time taken for the entire conversion process.
total_time = round(end - start, 2)
hours = total_time // 3600
minutes = (total_time % 3600) // 60
seconds = total_time % 60

# Print the total elapsed time in hours, minutes, and seconds.
print(f"Time taken: {hours} hours, {minutes} minutes and {seconds} seconds.")
