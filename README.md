# Agentic-Video-Summarizer-and-Quiz-Generator

This tool analyzes videos, generates detailed descriptions, stores them in a MongoDB database, and then creates multiple-choice quiz questions based on the video content.

## Overview

The application works in three main steps:
1. Analyzes a video file using Google's Gemini model
2. Stores the analysis in a MongoDB vector database
3. Generates quiz questions based on the video content using Groq's Qwen model

## Requirements

### Python Dependencies

```
pip install agno-core
```

The code requires the following Python packages (these will be installed as dependencies of agno-core):
- agno.media
- agno.models
- agno.document
- agno.vectordb
- agno.agent
- agno.embedder
- agno.tools
- agno.utils
- agno.knowledge

### External Services

1. **MongoDB**:
   - Running instance required (default connection: `mongodb://localhost:57572/?directConnection=true`)
   - Can be run via Docker (recommended, see Docker setup below)

2. **API Keys**:
   - Google Gemini API key (for video analysis and embedding)
   - Groq API key (for quiz generation)

## MongoDB Setup with Docker

1. Install Docker if not already installed
2. Run MongoDB container:
   ```bash
   docker run -d -p 57572:27017 --name video-summarizer-mongodb mongo:latest
   ```

## Environment Setup

Create a `.env` file in your project directory with the following (replace with your actual API keys):

```
GOOGLE_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

## How to Run

1. Ensure MongoDB is running
2. Make sure you have a video file ready for analysis
3. Run the script:
   ```bash
   python new.py
   ```
4. When prompted, enter the full path to your video file
5. Enter your prompt for video analysis (e.g., "Provide a detailed summary of this video")
6. Wait for the analysis and quiz generation to complete

## Output

The program will:
1. Display the video analysis results
2. Store the analysis in MongoDB
3. Generate and display 10-20 multiple-choice quiz questions based on the video content

## Troubleshooting

- If you encounter MongoDB connection issues, check that:
  - MongoDB is running on the specified port (57572)
  - Your firewall allows connections to this port
  - The connection string in the code matches your MongoDB setup

- For API-related errors:
  - Verify your API keys are correctly set
  - Check that you have sufficient quota/credits for the API services

## Customization

You can modify these parameters in the code:
- MongoDB collection name (`DB_COLLECTION_NAME`)
- MongoDB connection string (`MDB_CONNECTION_STRING`)
- Model selections (Gemini and Groq)
- Number of quiz questions (modify the quiz prompt)

## Note

This tool requires an active internet connection to access the Gemini and Groq APIs, as well as the DuckDuckGo search functionality.
