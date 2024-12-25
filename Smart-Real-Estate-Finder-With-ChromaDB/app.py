# Quick fix for RuntimeError: Your system has an unsupported version of sqlite3. Chroma  requires sqlite3 >= 3.35.0.
# import pysqlite3
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sys
import json
import os
import torch
import chromadb
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, VideoMAEModel, VideoMAEFeatureExtractor
import gradio as gr
import time
import cv2  # To extract frames from videos
import re

# Load the property data from the JSON file
with open("property_data.json", "r") as f:
    property_data = json.load(f)["properties"]

# Setup ChromaDB
client = chromadb.Client()
collection = client.create_collection("media_collection")  # Changed to general media collection

# Load CLIP model and processor for generating image and text embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load VideoMAE model and processor for video embeddings
video_model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
video_processor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base")

# Process images and generate embeddings
image_paths = [prop["image"] for prop in property_data]
images = [Image.open(image_path) for image_path in image_paths]
inputs = clip_processor(images=images, return_tensors="pt", padding=True)

# Measure image ingestion time
start_ingestion_time = time.time()

with torch.no_grad():
    image_embeddings = clip_model.get_image_features(**inputs).numpy()

# Convert numpy arrays to lists
image_embeddings = [embedding.tolist() for embedding in image_embeddings]

# Measure total ingestion time
end_ingestion_time = time.time()
ingestion_time = end_ingestion_time - start_ingestion_time

# Add image embeddings to the collection with metadata
collection.add(
    embeddings=image_embeddings,
    metadatas=[{
        "id": prop["id"], 
        "description": prop["description"],
        "price": prop["price"], 
        "address": prop["address"], 
        "features": ", ".join(prop["features"]),
        "image": prop["image"]
    } for prop in property_data],
    ids=[prop["id"] for prop in property_data]
)

# Log the ingestion performance
print(f"Image Data ingestion time: {ingestion_time:.4f} seconds")

# Function to extract a limited number of frames from a video
def extract_frames(video_path, num_frames=16):
    video_capture = cv2.VideoCapture(video_path)
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(total_frames // num_frames, 1)  # Interval to sample frames evenly
    frames = []
    for i in range(num_frames):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)  # Set the position for frame extraction
        ret, frame = video_capture.read()
        if not ret:
            break
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))
    video_capture.release()

    # If extracted frames are less than num_frames, pad with the last frame
    if len(frames) < num_frames:
        last_frame = frames[-1] if frames else Image.new("RGB", (224, 224), (0, 0, 0))  # Use black frame if none extracted
        frames.extend([last_frame] * (num_frames - len(frames)))

    return frames

# Function to get embeddings from video frames
def get_frame_embeddings(frames):
    inputs = clip_processor(images=frames, return_tensors="pt", padding=True)
    with torch.no_grad():
        frame_embeddings = clip_model.get_image_features(**inputs).numpy()
    return frame_embeddings.tolist()

# Function to calculate text similarity score based on description, address, price, and features
def get_text_similarity_score(query_text, property_data):
 
    """
    Calculate a similarity score based on query_text and the property data.
    Factors in description, address, price, and features.
    """
    score = 0

    # Normalize query text for comparison
    normalized_query = query_text.lower()

    # Check description
    description = property_data.get('description', '').lower()
    if normalized_query in description:
        score += 1.0  # Exact match in description

    # Check address
    address = property_data.get('address', '').lower()
    if normalized_query in address:
        score += 1.0  # Exact match in address

    # Check features (assuming features is a list of strings)
    features = property_data.get('features', [])
    for feature in features:
        if normalized_query in feature.lower():
            score += 0.5  # Partial match in features

    # Check price (assumes price is a string; adjust if it's numeric)
    price = property_data.get('price', '')
    if re.search(r'\b{}\b'.format(re.escape(normalized_query)), str(price)):
        score += 0.5  # Exact match in price (if the query text is a valid price string)

    return score

def search_properties(query_text, query_image, query_video):
    query_embedding = None
    text_embedding = None
    frame_embeddings = None

    # Generate an embedding for the query image if provided
    if query_image is not None:
        inputs = clip_processor(images=query_image, return_tensors="pt", padding=True)
        with torch.no_grad():
            query_embedding = clip_model.get_image_features(**inputs).numpy().flatten()

    # Generate an embedding for the query text if provided
    if query_text.strip():
        inputs = clip_processor(text=query_text, return_tensors="pt", padding=True)
        with torch.no_grad():
            text_embedding = clip_model.get_text_features(**inputs).numpy().flatten()

    # Generate embeddings for the query video if provided
    if query_video is not None:
        frames = extract_frames(query_video)  # Extract frames from the video
        frame_embeddings = get_frame_embeddings(frames)  # Get embeddings for the extracted frames
        if frame_embeddings:
            # Use the average of the frame embeddings for querying
            query_embedding = np.mean(frame_embeddings, axis=0)

    # Combine embeddings if both are provided
    if query_embedding is not None and text_embedding is not None:
        query_embedding = np.mean(np.array([query_embedding, text_embedding]), axis=0)
    elif query_embedding is None:  # If only text is provided
        query_embedding = text_embedding

    # Check if query_embedding is still None before proceeding
    if query_embedding is None:
        return None, "", []  # Return empty outputs if no embedding could be created

    # Perform a vector search in the collection
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=5
    )

    # Sort the results by distance (lowest to highest)
    sorted_results = sorted(
        zip(results['distances'][0], results['metadatas'][0]),
        key=lambda x: x[0]
    )

    # Separate sorted metadata and distances
    sorted_properties = [prop for _, prop in sorted_results]
    sorted_distances = [dist for dist, _ in sorted_results]

    properties_with_scores = []

    # If there's a text query, calculate text similarity scores
    if query_text.strip():
        # Sort the results with text similarity scores
        for prop in sorted_properties:
            score = get_text_similarity_score(query_text, prop)
            properties_with_scores.append((score, prop))
        
        # Sort properties based on similarity scores (highest first)
        properties_with_scores.sort(key=lambda x: x[0], reverse=True)
    else:
        # If no text query, just take the sorted results by vector distance
        properties_with_scores = [(None, prop) for prop in sorted_properties]

    # Prepare the best match image output
    best_match = properties_with_scores[0][1] if properties_with_scores else {}
    best_match_image_path = best_match.get('image')
    best_match_details = (
        f"Description: {best_match.get('description', '')}\n"
        f"Price: ${best_match.get('price', '')}\n"
        f"Address: {best_match.get('address', '')}"
    )

    # Prepare gallery data excluding the best match
    gallery_data = [
        {
            "image": prop[1]['image'],
            "text": f"Description: {prop[1].get('description', '')}, Price: ${prop[1].get('price', '')}, Address: {prop[1].get('address', '')}"
        }
        for prop in properties_with_scores[1:]  # Skip the best match
    ]

    gallery_data_list_format = [[prop['image'], prop['text']] for prop in gallery_data]
    return best_match_image_path, best_match_details, gallery_data_list_format


# Gradio Interface
iface = gr.Interface(
    fn=search_properties,
    inputs=[
        gr.Textbox(label="Enter your text query", placeholder="What are you looking for?"),
        gr.Image(type="pil", label="Upload an image for search (optional)"),
        gr.Video(label="Upload a video for search (optional)")
    ],
    outputs=[
        gr.Image(type="filepath", label="Best Match Image"),
        gr.Textbox(label="Best Match Details"),
        gr.Gallery(label="Similar Properties", columns=2, preview=True)
    ],
    title="Smart Real Estate Finder with Text, Image and Video Search",
    description="Upload an image, video, or enter a text query to find the best match and similar properties.",
    live=True
)

# Launch the Gradio app
iface.launch(debug=True)


# How to setup a custom ChromaDB path
# # Directory where the database will be stored
# db_dir = "db"

# # Check if the database directory exists; if not, create it
# if not os.path.exists(db_dir):
#     os.makedirs(db_dir)
#     print(f"Directory '{db_dir}' created.")
# else:
#     print(f"Directory '{db_dir}' already exists.")


# # Create a persistent Chroma client with new configuration
# client = chromadb.PersistentClient(path=db_dir)
