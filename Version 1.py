from pathlib import Path
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
    video_path = Path(__file__).parent.joinpath(user_video_path)
    if video_path.is_file():
        print(f"Video file found: {video_path}")
        break
    else:
        print(f"Error: File not found at '{user_video_path}'. Please check the path and try again.")

# Check if the video file exists
if not video_path.is_file():
    raise FileNotFoundError(f"Video file not found at: {video_path}. Please download a sample video or update the path.")

video = Video(
content=video_path.read_bytes(), # Load raw bytes
format="mp4" # Set the format explicitly
)

# MongoDB Configuration for Knowledge Base
MDB_CONNECTION_STRING = "mongodb://localhost:51083/?directConnection=true"
DB_COLLECTION_NAME = "Video_Summarization" 
# --- Main Code ---
# Step 1: Initialize Video Agent and Generate Description
print(f"--- Step 1: Analyzing Video ({video_path}) ---")
video_agent = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    markdown=True,
    tools=[DuckDuckGoTools()],
    use_json_mode=True,
    show_tool_calls=False
)

# Use agent.run() to get the description
video_description_prompt = f"""
    **Objective:** Analyze the provided video content and generate a structured Summary.

    **Based *only* on the provided video, please provide the following:**

    1.  **Main Topics Covered:**
        * Identify and list the primary subjects or themes discussed throughout the video. Use a concise bulleted list.

    2.  **Detailed Summary:**
        * Write a comprehensive summary that accurately captures the main arguments, information flow, key concepts, and conclusions presented in the video.
        * Structure the summary logically (e.g., following the video's progression or grouping related ideas).
        * Ensure the summary is detailed enough to give someone who hasn't seen the video a thorough understanding of its content.

    3.  **Key Takeaways / Highlights:**
        * Extract the most significant points, essential facts, core messages, findings, or actionable advice presented.
        * List these as clear, concise bullet points, focusing on the essential highlights.

    4.  **Relevant Online Resources:**
        * Identify and provide links to reputable online resources that directly expand on, support, or are referenced by the topics discussed in the video.
        * **Priority:** First, include any specific URLs or resource names explicitly mentioned in the video's content or accompanying description (if this information is available).
        * **Secondary:** If few or no resources are mentioned, find [Specify Number, e.g., 3-5] relevant, high-quality external links (e.g., official websites, reputable articles, academic papers, tools related to the topic).
        * **Annotation:** Briefly describe the content and relevance of each link provided.

    **Instructions:**
    * Use clear headings for each of the four sections requested above (e.g., "Main Topics Covered", "Detailed Summary", "Key Takeaways", "Relevant Online Resources").
    * Derive all summaries and key points *directly* from the provided video information. Do not add external information to the summary or key points.
    * Ensure the resource links are functional and relevant as of the current date.
"""

response: RunResponse = video_agent.run(
    video_description_prompt,
    videos=[video]
)

pprint_run_response(response, markdown=True)

convert_test = response.get_content_as_string()

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
    model=Groq(id="meta-llama/llama-4-scout-17b-16e-instruct"),
    knowledge=knowledge_base,
    show_tool_calls=True,
    
    )

# Ask the quiz agent to generate questions based on the knowledge base content
quiz_prompt = f"""
    Generate 10-20 multiple-choice quiz questions based *only* on the provided `Knowledge_base`.

    **Requirements:**

    * **Source:** Strictly use information *only* from the `Knowledge_base`. Ignore external data/references mentioned within it.
    * **Question Phrasing:** Frame questions naturally as standard quiz questions. **Do not use phrases** like "According to the knowledge base..." or similar source attributions.
    * **Content:** Focus questions on specific details found in the `Knowledge_base`: key facts, concepts, definitions, processes, steps, arguments, or specific terminology presented.
    * **Format:**
        * Numbered list for questions.
        * Each question: 1 correct answer, 3 plausible distractors (labeled A, B, C, D).
        * Clarity: Questions must be clear and directly answerable from the source.
    * **Output:** Indicate the correct answer by placing a tick emoji (✅) **directly next to the correct option**. No separate answer key needed.
"""

print(f"\n--- Asking Quiz Agent: '{quiz_prompt}' ---")
# Use print_response for direct output in this final step
quiz_agent.print_response(quiz_prompt, markdown=True)

