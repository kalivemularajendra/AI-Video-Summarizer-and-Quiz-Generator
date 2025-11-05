import streamlit as st
import tempfile
import os
import logging
import time
from pathlib import Path
from typing import Optional, Tuple
import hashlib
import traceback
from agno.media import Video
from agno.models.groq import Groq
from agno.models.google import Gemini
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.utils.pprint import pprint_run_response
import io
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
st.set_page_config(
    page_title="Video Analysis & Quiz Generator",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
MAX_FILE_SIZE_MB = 500  # Maximum file size in MB
SUPPORTED_VIDEO_FORMATS = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv', 'm4v']
SESSION_TIMEOUT_MINUTES = 30

# Initialize session state
def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'quiz_results' not in st.session_state:
        st.session_state.quiz_results = None
    if 'processed_file_hash' not in st.session_state:
        st.session_state.processed_file_hash = None
    if 'api_keys_validated' not in st.session_state:
        st.session_state.api_keys_validated = False
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = "idle"

def validate_api_keys(gemini_key: str, groq_key: str) -> Tuple[bool, str]:
    """Validate API keys format and basic structure"""
    try:
        if not gemini_key or not groq_key:
            return False, "Both API keys are required"
        
        # Basic validation for Gemini API key
        if not gemini_key.startswith('AI') or len(gemini_key) < 30:
            return False, "Invalid Gemini API key format"
        
        # Basic validation for Groq API key (they typically start with 'gsk_')
        if not groq_key.startswith('gsk_') or len(groq_key) < 40:
            return False, "Invalid Groq API key format"
        
        return True, "API keys validated successfully"
    except Exception as e:
        logger.error(f"Error validating API keys: {e}")
        return False, f"Error validating API keys: {str(e)}"

def get_file_hash(file_content: bytes) -> str:
    """Generate hash for file content to detect duplicates"""
    return hashlib.md5(file_content).hexdigest()

def validate_video_file(uploaded_file) -> Tuple[bool, str]:
    """Validate uploaded video file"""
    try:
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            return False, f"File size ({file_size_mb:.1f} MB) exceeds maximum allowed size ({MAX_FILE_SIZE_MB} MB)"
        
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in SUPPORTED_VIDEO_FORMATS:
            return False, f"Unsupported file format. Supported formats: {', '.join(SUPPORTED_VIDEO_FORMATS)}"
        
        # Check if file content exists
        if uploaded_file.size == 0:
            return False, "File is empty"
        
        return True, "File validation successful"
    except Exception as e:
        logger.error(f"Error validating video file: {e}")
        return False, f"Error validating file: {str(e)}"

@st.cache_data(ttl=3600)  # Cache for 1 hour
def cached_video_analysis(file_hash: str, content: bytes, api_key_hash: str) -> Optional[str]:
    """Cached video analysis to avoid reprocessing same files"""
    # This is a placeholder for caching logic
    # In a real implementation, you might use a database or file system cache
    return None

def safe_agent_execution(agent_func, *args, **kwargs):
    """Safely execute agent functions with proper error handling"""
    try:
        return agent_func(*args, **kwargs)
    except Exception as e:
        logger.error(f"Agent execution error: {e}")
        logger.error(traceback.format_exc())
        raise

def cleanup_temp_files(file_paths: list):
    """Safely cleanup temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to cleanup file {file_path}: {e}")

# Initialize session state
initialize_session_state()

st.title("🎥 Video Analysis & Quiz Generator")
st.markdown("Upload a video file and get AI-powered analysis with quiz questions!")

# Add usage statistics
if st.checkbox("📊 Show Usage Tips", value=False):
    st.info("""
    **💡 Tips for better results:**
    - Use videos with clear audio and speech
    - Educational or informational videos work best
    - Optimal file size: 10-100 MB
    - Supported formats: MP4, AVI, MOV, MKV, WEBM, FLV, M4V
    """)

# Sidebar for API keys
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("Enter your API keys to use the application:")
    
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Get your API key from Google AI Studio",
        placeholder="AIza..."
    )
    
    groq_api_key = st.text_input(
        "Groq API Key", 
        type="password",
        help="Get your API key from Groq Console",
        placeholder="gsk_..."
    )
    
    # Validate API keys
    if gemini_api_key and groq_api_key:
        is_valid, message = validate_api_keys(gemini_api_key, groq_api_key)
        if is_valid:
            st.success("✅ API keys validated")
            st.session_state.api_keys_validated = True
        else:
            st.error(f"❌ {message}")
            st.session_state.api_keys_validated = False
    else:
        st.session_state.api_keys_validated = False
    
    st.markdown("---")
    st.markdown("### 📋 Instructions")
    st.markdown("""
    1. Enter your API keys above
    2. Upload a video file
    3. Click 'Analyze Video' to get analysis
    4. View the generated quiz questions
    """)
    
    # System status
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    if st.session_state.processing_status == "processing":
        st.warning("⚠️ Processing in progress...")
    elif st.session_state.analysis_results:
        st.success("✅ Analysis complete")
    else:
        st.info("💤 Ready for analysis")

# Check if API keys are provided and validated
if not st.session_state.api_keys_validated:
    st.warning("⚠️ Please enter valid Gemini and Groq API keys in the sidebar to continue.")
    st.stop()

# Set environment variables for API keys (only if validated)
try:
    os.environ["GEMINI_API_KEY"] = gemini_api_key
    os.environ["GROQ_API_KEY"] = groq_api_key
except Exception as e:
    st.error(f"❌ Error setting API keys: {e}")
    st.stop()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📁 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_VIDEO_FORMATS,
        help=f"Upload a video file to analyze (Max: {MAX_FILE_SIZE_MB}MB)"
    )
    
    # File validation
    if uploaded_file is not None:
        is_valid_file, validation_message = validate_video_file(uploaded_file)
        if not is_valid_file:
            st.error(f"❌ {validation_message}")
            st.stop()
        else:
            st.success(f"✅ {validation_message}")

# Show cached results if available
if uploaded_file is not None:
    file_hash = get_file_hash(uploaded_file.getvalue())
    
    # Check if this file was already processed
    if (st.session_state.processed_file_hash == file_hash and 
        st.session_state.analysis_results and 
        st.session_state.quiz_results):
        
        with col2:
            st.header("📹 Video Preview")
            st.video(uploaded_file)
        
        st.info("📊 Using cached results for this file. Click 'Reprocess Video' to analyze again.")
        
        # Display cached results
        st.header("📋 Video Analysis Results (Cached)")
        st.markdown(st.session_state.analysis_results)
        
        st.header("❓ Generated Quiz Questions (Cached)")
        st.markdown(st.session_state.quiz_results)
        
        # Option to reprocess
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Reprocess Video", type="secondary", use_container_width=True):
                st.session_state.analysis_results = None
                st.session_state.quiz_results = None
                st.session_state.processed_file_hash = None
                st.rerun()
        
        # Download options for cached results
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="📥 Download Analysis",
                data=st.session_state.analysis_results,
                file_name=f"video_analysis_{uploaded_file.name}.md",
                mime="text/markdown"
            )
        
        with col2:
            st.download_button(
                label="📥 Download Quiz",
                data=st.session_state.quiz_results,
                file_name=f"quiz_{uploaded_file.name}.md",
                mime="text/markdown"
            )
        st.stop()

if uploaded_file is not None:
    with col2:
        st.header("📹 Video Preview")
        try:
            st.video(uploaded_file)
        except Exception as e:
            st.error(f"❌ Could not preview video: {e}")
    
    # Display file info
    file_size_mb = uploaded_file.size / (1024*1024)
    st.info(f"📊 File: {uploaded_file.name} | Size: {file_size_mb:.2f} MB | Format: {uploaded_file.name.split('.')[-1].upper()}")
    
    # Processing button with state management
    button_disabled = st.session_state.processing_status == "processing"
    button_text = "⏳ Processing..." if button_disabled else "🔍 Analyze Video"
    
    if st.button(button_text, type="primary", use_container_width=True, disabled=button_disabled):
        # Set processing state
        st.session_state.processing_status = "processing"
        
        # Initialize containers for real-time updates
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
        
        with status_container:
            status_text = st.empty()
        
        temp_files = []  # Track temporary files for cleanup
        
        try:
            # Save uploaded file temporarily
            status_text.info("💾 Processing video file...")
            progress_bar.progress(5)
            
            file_extension = uploaded_file.name.split('.')[-1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                video_path = tmp_file.name
                temp_files.append(video_path)
            
            # Load video content with error handling
            status_text.info("📖 Loading video content...")
            progress_bar.progress(15)
            
            try:
                with open(video_path, "rb") as f:
                    video_bytes = f.read()
                
                if not video_bytes:
                    raise ValueError("The video file is empty after reading.")
                    
                video_obj = Video(content=video_bytes)
                
            except Exception as e:
                st.error(f"❌ Failed to load video content: {e}")
                cleanup_temp_files(temp_files)
                st.session_state.processing_status = "idle"
                st.stop()
            
            # Step 1: Video Analysis with retry logic
            status_text.info("🤖 Initializing video analysis...")
            progress_bar.progress(25)
            
            max_retries = 3
            retry_count = 0
            analysis_success = False
            
            while retry_count < max_retries and not analysis_success:
                try:
                    video_agent = Agent(
                        model=Gemini(id="gemini-2.0-flash-lite"),
                        markdown=True,
                        tools=[DuckDuckGoTools()],
                        use_json_mode=True,
                        show_tool_calls=False
                    )
                    
                    video_description_prompt = """
                    **Objective:** Analyze the provided video content (e.g., transcript, description) and generate a structured breakdown.

                    **Based *only* on the provided information about the video, please provide the following:**

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
                        * **Secondary:** If few or no resources are mentioned, find 3-5 relevant, high-quality external links (e.g., official websites, reputable articles, academic papers, tools related to the topic).
                        * **Annotation:** Briefly describe the content and relevance of each link provided.

                    **Instructions:**
                    * Use clear headings for each of the four sections requested above (e.g., "Main Topics Covered", "Detailed Summary", "Key Takeaways", "Relevant Online Resources").
                    * Derive all summaries and key points *directly* from the provided video information. Do not add external information to the summary or key points.
                    * Ensure the resource links are functional and relevant as of the current date.
                    """
                    
                    status_text.info(f"🔍 Analyzing video content... (Attempt {retry_count + 1}/{max_retries})")
                    progress_bar.progress(40 + (retry_count * 10))
                    
                    # Execute analysis with timeout protection
                    start_time = time.time()
                    response: RunResponse = safe_agent_execution(
                        video_agent.run,
                        video_description_prompt,
                        videos=[video_obj]
                    )
                    end_time = time.time()
                    
                    # Extract the content from the response
                    if hasattr(response, 'content') and response.content:
                        knowledge_text = response.content
                    else:
                        knowledge_text = str(response)
                    
                    if not knowledge_text or len(knowledge_text.strip()) < 50:
                        raise ValueError("Analysis returned insufficient content")
                    
                    analysis_success = True
                    logger.info(f"Video analysis completed in {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Analysis attempt {retry_count} failed: {e}")
                    if retry_count >= max_retries:
                        st.error(f"❌ Video analysis failed after {max_retries} attempts: {e}")
                        cleanup_temp_files(temp_files)
                        st.session_state.processing_status = "idle"
                        st.stop()
                    
                    # Wait before retry
                    time.sleep(2)
            
            # Store analysis results in session state
            st.session_state.analysis_results = knowledge_text
            progress_bar.progress(60)
            
            # Display video analysis results
            status_text.success("✅ Video analysis completed!")
            with st.expander("� Video Analysis Results", expanded=True):
                st.markdown(knowledge_text)
            
            # Step 2: Generate Quiz Questions with retry logic
            status_text.info("❓ Generating quiz questions...")
            progress_bar.progress(70)
            
            retry_count = 0
            quiz_success = False
            
            while retry_count < max_retries and not quiz_success:
                try:
                    quiz_agent = Agent(
                        model=Groq(id="llama-3.3-70b-versatile"),
                        show_tool_calls=False
                    )
                    
                    quiz_prompt = f"""
                    Based *only* on the following text extracted from the video description:

                    --- START VIDEO DESCRIPTION TEXT ---
                    {knowledge_text}
                    --- END VIDEO DESCRIPTION TEXT ---

                    Generate 10-20 multiple-choice quiz questions based *only* on the text provided above between the 'START' and 'END' markers.

                    **Requirements:**

                    * **Source:** Strictly use information *only* from the text provided above. Ignore external data/references mentioned within it.
                    * **Question Phrasing:** Frame questions naturally as standard quiz questions. **Do not use phrases** like "According to the knowledge base..." or "Based on the text..." or similar source attributions within the question itself.
                    * **Content:** Focus questions on specific details found in the provided text: key facts, concepts, definitions, processes, steps, arguments, or specific terminology presented.
                    * **Format:**
                        * Numbered list for questions (1., 2., 3., ...).
                        * Each question: 1 correct answer, 3 plausible distractors (labeled A, B, C, D).
                        * Clarity: Questions must be clear and directly answerable from the source text provided above.
                        * Example Format:
                            1.  What is the primary subject of the video?
                                A. Subject A
                                B. Subject B
                                C. Subject C ✅
                                D. Subject D
                    * **Output:** Indicate the correct answer by placing a tick emoji (✅) **directly next to the correct option**. No separate answer key needed.
                    """
                    
                    status_text.info(f"🎯 Generating quiz... (Attempt {retry_count + 1}/{max_retries})")
                    progress_bar.progress(80 + (retry_count * 5))
                    
                    # Execute quiz generation
                    start_time = time.time()
                    quiz_response = safe_agent_execution(quiz_agent.run, quiz_prompt)
                    end_time = time.time()
                    
                    quiz_content = quiz_response.content if hasattr(quiz_response, 'content') else str(quiz_response)
                    
                    if not quiz_content or len(quiz_content.strip()) < 100:
                        raise ValueError("Quiz generation returned insufficient content")
                    
                    quiz_success = True
                    logger.info(f"Quiz generation completed in {end_time - start_time:.2f} seconds")
                    
                except Exception as e:
                    retry_count += 1
                    logger.error(f"Quiz generation attempt {retry_count} failed: {e}")
                    if retry_count >= max_retries:
                        st.error(f"❌ Quiz generation failed after {max_retries} attempts: {e}")
                        cleanup_temp_files(temp_files)
                        st.session_state.processing_status = "idle"
                        st.stop()
                    
                    # Wait before retry
                    time.sleep(2)
            
            # Store quiz results in session state
            st.session_state.quiz_results = quiz_content
            st.session_state.processed_file_hash = file_hash
            progress_bar.progress(95)
            
            # Display quiz questions
            status_text.success("✅ Quiz generation completed!")
            with st.expander("❓ Generated Quiz Questions", expanded=True):
                st.markdown(quiz_content)
            
            # Cleanup temporary files
            cleanup_temp_files(temp_files)
            
            # Final progress update
            progress_bar.progress(100)
            status_text.success("🎉 Video analysis and quiz generation completed successfully!")
            
            # Reset processing status
            st.session_state.processing_status = "idle"
            
            # Download options
            st.markdown("---")
            st.subheader("📥 Download Results")
            col1, col2 = st.columns(2)
            with col1:
                st.download_button(
                    label="📄 Download Analysis",
                    data=knowledge_text,
                    file_name=f"video_analysis_{uploaded_file.name}_{int(time.time())}.md",
                    mime="text/markdown",
                    help="Download the video analysis as a markdown file"
                )
            
            with col2:
                st.download_button(
                    label="❓ Download Quiz",
                    data=quiz_content,
                    file_name=f"quiz_{uploaded_file.name}_{int(time.time())}.md",
                    mime="text/markdown",
                    help="Download the quiz questions as a markdown file"
                )
                
        except Exception as e:
            # Comprehensive error handling
            error_message = str(e)
            st.error(f"❌ An unexpected error occurred: {error_message}")
            
            # Log the full traceback for debugging
            logger.error(f"Unexpected error in video processing: {e}")
            logger.error(traceback.format_exc())
            
            # Show detailed error in expander for debugging
            with st.expander("🔍 Error Details (for debugging)", expanded=False):
                st.code(traceback.format_exc())
            
            # Cleanup on error
            cleanup_temp_files(temp_files)
            
            # Reset processing status
            st.session_state.processing_status = "idle"
            
            # Suggest troubleshooting steps
            st.info("""
            **💡 Troubleshooting suggestions:**
            - Check your internet connection
            - Verify your API keys are valid and have sufficient credits
            - Try with a smaller video file
            - Ensure the video format is supported
            - Contact support if the issue persists
            """)

else:
    st.info("👆 Please upload a video file to begin analysis.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
        <p>Built with ❤️ using Streamlit, Agno, Gemini, and Groq</p>
        <p>Version 2.0 - Enhanced with robust error handling and caching</p>
    </div>
    """, 
    unsafe_allow_html=True
)

