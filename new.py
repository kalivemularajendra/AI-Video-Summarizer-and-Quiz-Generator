from pathlib import Path
from typing import Iterator
from agno.media import Video
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.document.base import Document
from agno.vectordb.mongodb import MongoDb
from agno.agent import Agent, RunResponse
from agno.embedder.google import GeminiEmbedder
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.pprint import pprint_run_response
from agno.knowledge.document import DocumentKnowledgeBase



while True:
    user_video_path = input("Please enter the full path to the video file: ")
    video_path = Path(user_video_path)
    if video_path.is_file():
        print(f"Video file found: {video_path}")
        break
    else:
        print(f"Error: File not found at '{user_video_path}'. Please check the path and try again.")

video_object = Video(filepath=video_path)

# Check if the video file exists
if not video_path.is_file():
    raise FileNotFoundError(f"Video file not found at: {video_path}. Please download a sample video or update the path.")

# MongoDB Configuration for Knowledge Base
MDB_CONNECTION_STRING = "mongodb://localhost:57572/?directConnection=true"
DB_COLLECTION_NAME = "Video_Summarization" 
# --- Main Code ---
# Step 1: Initialize Video Agent and Generate Description
print(f"--- Step 1: Analyzing Video ({video_path}) ---")
video_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    markdown=True,
    tools=[DuckDuckGoTools()],
    use_json_mode=True
)

# Use agent.run() to get the description
video_description_prompt = input("Please enter the prompt to Analyze the video: ")
response: RunResponse = video_agent.run(
    video_description_prompt,
    videos=[video_object]
)

pprint_run_response(response, markdown=True)

convert_test = str(response)

# Step 2: Store Description in MongoDB Knowledge Base
print(f"\n--- Step 2: Storing Description in MongoDB (Collection: {DB_COLLECTION_NAME}) ---")

# Create a Document object from the generated text
description_document = Document(content=convert_test)

# Initialize the MongoDB vector store
mongo_vector_db = MongoDb(
    collection_name=DB_COLLECTION_NAME,
    db_url=MDB_CONNECTION_STRING,
    wait_until_index_ready=60, 
    wait_after_insert=5, 
    embedder=GeminiEmbedder(dimensions=1536), 
)

# Create a Document Knowledge Base using the description document and MongoDB
# Use a list containing the single document
knowledge_base = DocumentKnowledgeBase(
    documents=[description_document],
    vector_db=mongo_vector_db,
)

# Load the knowledge base into MongoDB (recreate=True will delete existing data in the collection)
try:
    print("Loading knowledge base into MongoDB...")
    knowledge_base.load(recreate=True)
    print("Knowledge base loaded successfully.")
except Exception as e:
    print(f"Error loading knowledge base into MongoDB: {e}")
    exit() 


# Step 3: Initialize Quiz Agent and Generate Questions
print("\n--- Step 3: Generating Quiz Questions based on Video Description ---")

# Create the quiz agent, providing the knowledge base
quiz_agent = Agent(
    model=Groq(id="qwen-qwq-32b"),
    knowledge=knowledge_base,
)

# Ask the quiz agent to generate questions based on the knowledge base content
quiz_prompt = f"Based *only* on the Knowledge_base, generate 10-20 multiple-choice quiz questions with answers. Exclue Questions related to Online references provided in the video description. The questions should be based on the content of the video and should not include any external references. The questions should be clear and concise, and the answers should be provided in a separate list. The questions should be suitable for a quiz format, with one correct answer and three distractors for each question. Please provide the questions in a numbered list format."

print(f"\n--- Asking Quiz Agent: '{quiz_prompt}' ---")
# Use print_response for direct output in this final step
quiz_agent.print_response(quiz_prompt, markdown=True)

