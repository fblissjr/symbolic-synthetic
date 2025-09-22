# run_heylook_examples.py
import os
import openai
import requests
import base64
import json
import re
import argparse
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# --- Configuration ---
# Point this to your heylookitsanllm server's OpenAI-compatible endpoint
BASE_URL = "http://localhost:8080/v1"
API_KEY = "not-needed" # API key is not required for local server

# Use model names from your models.yaml.example
TEXT_MODEL = "magistral"
VISION_MODEL = "magistral"

# --- Initialize OpenAI Client ---
client = openai.OpenAI(base_url=BASE_URL, api_key=API_KEY)

# --- Helper Functions ---
def create_dummy_image(text: str, size: tuple = (256, 256), bg_color="blue") -> bytes:
    img = Image.new('RGB', size, color=bg_color)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill="white", font=font)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()

def image_bytes_to_base64_url(image_bytes: bytes) -> str:
    encoded = base64.b64encode(image_bytes).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def check_server_capabilities():
    print("--- 1. Checking Server Capabilities ---")
    try:
        response = requests.get(f"{BASE_URL.replace('/v1', '')}/v1/capabilities")
        response.raise_for_status()
        capabilities = response.json()
        print("✅ Server capabilities fetched successfully.")
        print(f"  Fast vision endpoint available: {capabilities['endpoints']['fast_vision']['available']}")
        print(f"  Available optimizations: {list(capabilities['optimizations'].keys())}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error checking capabilities: {e}")
    print("-" * 35 + "\n")

def simple_text_chat():
    # (Implementation is the same as before)
    print("--- 2. Simple Text Chat Completion ---")
    try:
        response = client.chat.completions.create(
            model=TEXT_MODEL, messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Explain the concept of an LRU cache in one sentence."}], max_tokens=100
        )
        print(f"✅ Response from '{TEXT_MODEL}':")
        print(response.choices[0].message.content)
    except openai.APIConnectionError:
        print(f"❌ Connection Error: Is the server running at {BASE_URL}?")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    print("-" * 35 + "\n")

def streaming_chat():
    print("--- 3. Streaming Text Chat Completion ---")
    try:
        stream = client.chat.completions.create(
            model=TEXT_MODEL,
            messages=[{"role": "user", "content": "Write a short poem about local LLMs."}],
            stream=True,
            max_tokens=150
        )
        print(f"✅ Streaming response from '{TEXT_MODEL}':")
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        print()
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    print("-" * 35 + "\n")

def vision_chat_base64():
    print("--- 4. Vision Chat (Standard Base64) ---")
    try:
        image_bytes = create_dummy_image("Image #1", bg_color="red")
        base64_url = image_bytes_to_base64_url(image_bytes)
        response = client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": "What color is this image and what text does it contain?"},
                    {"type": "image_url", "image_url": {"url": base64_url}}
                ]}
            ],
            max_tokens=50
        )
        print(f"✅ Response from '{VISION_MODEL}':")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    print("-" * 35 + "\n")

def fast_vision_chat_multipart():
    # NOTE: This function remains the same, but we expect it to fail
    # due to the server-side bug identified above.
    print("--- 5. FAST Vision Chat (Multipart Upload) ---")
    print("   (Expecting server-side error due to multi-image handling bug)")
    try:
        image1_bytes = create_dummy_image("First Image", bg_color="purple")
        image2_bytes = create_dummy_image("Second Image", bg_color="green")
        messages_payload = json.dumps([
            {"role": "user", "content": [
                {"type": "text", "text": "Describe the two images provided."},
                {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}},
                {"type": "image_url", "image_url": {"url": "__RAW_IMAGE__"}}
            ]}
        ])
        files = {'images': ('image1.png', image1_bytes, 'image/png'), 'images': ('image2.png', image2_bytes, 'image/png')}
        data = {'model': VISION_MODEL, 'messages': messages_payload, 'max_tokens': 100}
        response = requests.post(f"{BASE_URL}/chat/completions/multipart", files=files, data=data)
        response.raise_for_status()
        result = response.json()

        # Check if the server returned an error in the response content
        if "error" in result['choices'][0]['message']['content'].lower():
            print(f"✅ Server correctly returned an error as expected:")
            print(result['choices'][0]['message']['content'])
        else:
            print(f"✅ Response from '{VISION_MODEL}' (via multipart):")
            print(result['choices'][0]['message']['content'])

    except Exception as e:
        print(f"❌ An error occurred: {e}")
    print("-" * 35 + "\n")

def generate_embeddings():
    print("--- 6. Generate REAL Model Embeddings ---")
    try:
        texts_to_embed = [
            "The quick brown fox jumps over the lazy dog.",
            "Local LLMs are changing the AI landscape.",
            "Inference can be performed on-device."
        ]
        response = client.embeddings.create(input=texts_to_embed, model=TEXT_MODEL)
        first_embedding = response.data[0].embedding
        print(f"✅ Successfully generated embeddings with model '{TEXT_MODEL}'.")
        print(f"  Number of embeddings: {len(response.data)}")
        print(f"  Dimensions of first embedding: {len(first_embedding)}")
        print(f"  First 5 dimensions: {first_embedding[:5]}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")
    print("-" * 35 + "\n")

def batch_processing_example():
    """Demonstrates batch processing by manually sending a request."""
    print("--- 7. Batch Processing (Multiple Conversations) ---")
    try:
        # Define the separate logical "conversations" or prompts
        batch_messages = [
            # First conversation
            {"role": "user", "content": "What is the capital of Japan?"},
            {"role": "system", "content": "___CONVERSATION_BOUNDARY___"}, # heylookitsanllm delimiter
            # Second conversation
            {"role": "user", "content": "What is the capital of Canada?"}
        ]

        # Manually construct the request payload with custom parameters
        payload = {
            "model": TEXT_MODEL,
            "messages": batch_messages,
            "processing_mode": "sequential", # Custom heylookitsanllm parameter
            "return_individual": True,       # Custom heylookitsanllm parameter
            "temperature": 0.5
        }

        # Use `requests` library to send the POST request with the custom JSON body
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload)
        response.raise_for_status() # Raise an exception for bad status codes

        result = response.json()

        # The server returns a custom "completions" list for individual batch responses
        completions = result.get("completions", [])

        print(f"✅ Received {len(completions)} separate responses from batch request:")
        for i, completion in enumerate(completions):
            print(f"  Response #{i+1}: {completion['choices'][0]['message']['content'].strip()}")

    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print("   Note: Ensure your server supports the 'processing_mode' and 'return_individual' parameters.")
    print("-" * 35 + "\n")

def chunk_document(markdown_text: str) -> list[str]:
    """Splits the document by double newlines."""
    return [chunk for chunk in re.split(r'\n\n+', markdown_text) if chunk.strip()]

SINGLE_CHUNK_PROMPT_TEMPLATE = """
You are an expert AI data preprocessor. Your task is to analyze a single chunk of a document and classify its pedagogical purpose.

Respond with ONLY a single, valid JSON object with the following keys: "id", "node_type", "title", and "content".

- `id`: Use the provided chunk ID.
- `node_type`: Classify into "CONCEPT", "WORKED_EXAMPLE", "PRACTICE_PROBLEM", or "UNKNOWN".
- `title`: Create a concise, 5-10 word title summarizing the content.
- `content`: Include the full, unmodified text content of the chunk.

Here is the chunk to classify:

---
CHUNK_ID: {chunk_id}
CONTENT:
{chunk_content}
---
"""

def perform_semantic_segmentation(filepath: str):
    """
    Reads a markdown file, chunks it, and uses the server's batch API
    to classify each chunk into a structured format.
    """
    print(f"--- 8. Stage 1 KPG Generation (Batch Mode) on '{filepath}' ---")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print(f"❌ Error: File not found at '{filepath}'")
        print("-" * 35 + "\n")
        return

    chunks = chunk_document(text)
    print(f"  📄 Document split into {len(chunks)} chunks.")

    batch_messages = []
    for i, chunk in enumerate(chunks):
        # Create the specific instruction for this chunk
        prompt_content = SINGLE_CHUNK_PROMPT_TEMPLATE.format(
            chunk_id=f"chunk_{i}",
            chunk_content=chunk
        )
        batch_messages.append({"role": "user", "content": prompt_content})

        # Add the special delimiter to tell the server this is a separate inference
        if i < len(chunks) - 1:
            batch_messages.append({"role": "system", "content": "___CONVERSATION_BOUNDARY___"})

    # Manually construct the request payload with custom batch parameters
    payload = {
        "model": TEXT_MODEL,
        "messages": batch_messages,
        "processing_mode": "sequential",  # Custom heylookitsanllm parameter
        "return_individual": True,        # Custom heylookitsanllm parameter
        "temperature": 0.0 # Use low temp for deterministic classification
    }

    print(f"  📦 Sending a single batch request for all {len(chunks)} chunks...")

    try:
        response = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=300) # Long timeout for big files
        response.raise_for_status()

        result = response.json()
        completions = result.get("completions", [])

        print(f"  ✅ Server processed the batch and returned {len(completions)} individual responses.")

        structured_nodes = []
        for i, completion in enumerate(completions):
            try:
                # The content of the response should be our desired JSON
                content_str = completion['choices'][0]['message']['content']
                # Clean potential markdown code fences
                if content_str.strip().startswith("```json"):
                    content_str = content_str.strip()[7:-3]

                node = json.loads(content_str)
                structured_nodes.append(node)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"  ⚠️ Warning: Could not parse response for chunk {i}: {e}")
                print(f"     Raw content: {completion['choices'][0]['message']['content']}")

        print(f"  📊 Successfully parsed {len(structured_nodes)} nodes.")
        print("  --- Sample of Structured Output ---")
        print(json.dumps(structured_nodes[:3], indent=2)) # Print first 3 nodes as a sample
        print("  ...")

    except requests.exceptions.RequestException as e:
        print(f"❌ An error occurred during the batch request: {e}")

    print("-" * 35 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run examples against heylookitsanllm server.")
    parser.add_argument(
        '--file',
        type=str,
        help="Optional path to a Markdown file to run the semantic segmentation on."
    )
    args = parser.parse_args()

    print("🚀 Running examples against heylookitsanllm server...")
    print(f"   Targeting API Base: {BASE_URL}\n")

    check_server_capabilities()
    simple_text_chat()
    streaming_chat()
    vision_chat_base64()
    fast_vision_chat_multipart()
    generate_embeddings()
    batch_processing_example()

    print("🎉 All examples complete.")
