
# Smart Real Estate Finder AI with Text, Image and Video Search

## Overview
The **Smart Real Estate Finder AI** is an innovative application that allows users to find properties based on various inputs, including text queries, images, videos or a combination of them. The application utilizes advanced AI models for image and video processing and ChromaDB as the vector search database ensuring accurate and relevant results.

## Demo Link
The Demo for this project can be found [here](https://youtu.be/TsxPzFe4fMc).


## Features
- **Multi-Modal Search**: Search properties using text queries, images, videos or a combination.
- **Real-Time Results**: Get instant property matches and details.
- **User-Friendly Interface**: Easy-to-navigate interface designed for optimal user experience.
- **Gallery View**: View similar properties in a gallery format.


## Installation and Setup

To run this project locally, follow these steps:

 
### Clone the Repository

```bash
git clone https://github.com/rudye223/Smart-Real-Estate-Finder-AI-With-ChromaDB
cd  Smart-Real-Estate-Finder-With-ChromaDB

```
### Setup
Create a Python virtual environment to manage the project's dependencies by running the following command on your terminal:
```bash 
 python3 -m venv .venv
```

Now, activate the virtual environment:

On macOS/ Linux:
```bash
source .venv/bin/activate
```

On Windows:
```bash
.venv\Scripts\activate
```

With the virtual environment activated, install the necessary dependencies by running this command:
```bash
 
pip install -r requirements.txt
```
Launch the application by running the script:
```bash
python3 app.py
```


## Requirements
- Python 3.7 or higher
- Required packages:
  - `torch`
  - `transformers`
  - `PIL`
  - `gradio`
  - `opencv-python`
  - `chromadb`
  - `numpy`
  - `pysqlite3`


Add as many property data as you want to the property_data.json. Ensure you have a JSON file (property_data.json) containing your property data in the following format:

```json
 
{
    "properties": [
        {
            "id": "1",
            "description": "A beautiful house in the countryside.",
            "price": "300000",
            "address": "123 Country Lane",
            "features": ["garden", "garage", "swimming pool"],
            "image": "path/to/image1.jpg"
        },
        ...
    ]
}
```

## How to Use the Application

### Accessing the Application

Launch the application by navigating to the provided local URL after running the application script.


### 1. Text Search
#### How It Works
- **Embedding Process**: The system utilizes a model like OpenAI's CLIP to generate a vector (embedding) from the provided text. This model understands the semantics of the query, identifying keywords such as property types, locations, and features (e.g., "luxury apartment with sea view").
- **Vector Search**: The generated text embedding is compared with precomputed embeddings of property data stored in ChromaDB, which is optimized for vector searches.
- **Result Matching**: The system retrieves the most similar properties based on vector proximity, displaying the closest match and up to four other similar properties.

#### Example Queries
- "2-bedroom apartment in downtown"
- "Luxury villa with ocean view"
- "Pet-friendly house with garden"

### 2. Image Search
#### How It Works
- **Image Embedding**: The system extracts visual features from the uploaded image using CLIP or similar vision-based models, converting the image into a vector embedding. This embedding captures visual aspects like colors, architecture styles, and overall design.
- **Image Matching**: The generated image embedding is compared to stored property embeddings in the dataset.
- **Result Matching**: The system retrieves properties that visually match the uploaded image, presenting the closest visual match and up to four similar properties.

#### Example Use Cases
- Upload a picture of a modern high-rise building to find similar apartments.
- Upload an image of a house brochure to locate properties with similar architecture or design.

### 3. Video Search
#### How It Works
- **Frame Extraction**: Key frames are extracted from the video at intervals, capturing different perspectives of the property.
- **Frame Embedding**: Each frame is converted into an embedding using VideoMAE or another video understanding model. These embeddings represent the property's visual features over time.
- **Video Matching**: The frame embeddings are compared to stored property embeddings.
- **Result Matching**: The system retrieves properties that match the video's overall style, returning the best match and up to four similar properties.

#### Example Use Cases
- Upload a video tour of a beachfront property to find similar beachside homes.
- Upload a video showcasing a property's neighborhood to find homes in visually similar areas.

## Combining Queries

### Text + Image Search
- The system combines the embeddings from both text and image inputs, refining search results based on both criteria.
#### Example:
- Search for "Modern apartment" with an image of a similar apartment.

### Text + Video Search
- The system combines text and video embeddings to find properties that match both the textual description and the visual style.
#### Example:
- Search for "House with pool" while uploading a video tour of a house with a pool.

### Image + Video Search
- The system combines image and video embeddings to refine search results, matching both visual inputs.
#### Example:
- Upload a photo and a video of a property to search for homes that match both.

### Text + Image + Video Search
- The system processes all three inputs together, prioritizing properties that match the text query, image, and video.
#### Example:
- Search for "3-bedroom house" while uploading an image and a video showcasing the house.

## Query Optimization

- **Best Match Image**: The property image that most closely aligns with the search criteria (text, image, or video) is displayed.
- **Similar Properties**: Up to four visually or semantically similar properties are presented for comparison.



```bash
#error with chroma  workaround 
# first: pip install pysqlite3-binary
# then add the code below in your py file:
# these three lines swap the stdlib sqlite3 lib with the pysqlite3 
import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

```
```bash
Also if you run into cv2 ImportError: libGL.so.1: cannot open shared object file: No such file or directory:
sudo apt update
sudo apt install libgl1-mesa-glx

```
## Acknowledgments

This project makes use of the following technologies:

- **Hugging Face Transformers**
- **OpenAI CLIP**
- **VideoMAE**

 
