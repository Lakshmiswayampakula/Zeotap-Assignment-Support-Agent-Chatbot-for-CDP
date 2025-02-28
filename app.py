import streamlit as st
import os
import tempfile
import requests
from typing import List
from dotenv import load_dotenv

from agno.agent import Agent
from agno.document import Document
from agno.models.google import Gemini
from agno.utils.log import logger
from agno.utils.pprint import pprint_run_response

from agentic_rag import get_cdp_support_agent  # Modified function name to match CDP context

load_dotenv()

st.set_page_config(
    page_title="CDP Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for app styling - Updated with CDP theme
CUSTOM_CSS = """
    <style>
    /* Dark Theme Styles with CDP color scheme */
    body {
        background-color: #0F172A;
        color: #E2E8F0;
    }

    /* Main Titles */
    .main-title {
        text-align: center;
        background: linear-gradient(45deg, #3B82F6, #60A5FA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3em;
        font-weight: bold;
        padding: 0.8em 0 0.4em 0;
    }
    
    .subtitle {
        text-align: center;
        color: #94A3B8;
        margin-bottom: 1.5em;
        font-size: 1.2em;
    }

    /* CDP Platform Pills */
    .cdp-platforms {
        display: flex;
        justify-content: center;
        gap: 10px;
        margin-bottom: 2em;
    }
    
    .cdp-pill {
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
    }
    
    .segment-pill {
        background-color: #52BD95;
        color: white;
    }
    
    .mparticle-pill {
        background-color: #5271FF;
        color: white;
    }
    
    .lytics-pill {
        background-color: #F59E0B;
        color: white;
    }
    
    .zeotap-pill {
        background-color: #EC4899;
        color: white;
    }

    /* Common button styling */
    .stButton button, div[data-testid="stDownloadButton"] button {
        width: 100%;
        border-radius: 15px;
        background-color: rgba(59, 130, 246, 0.2);
        color: white;
        border: 1px solid rgba(59, 130, 246, 0.3);
        transition: all 0.3s ease;
        padding: 10px;
        font-size: 16px;
        cursor: pointer;
    }

    /* Consistent glow effect for all buttons */
    .stButton button:hover, div[data-testid="stDownloadButton"] button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.6);
        background-color: rgba(59, 130, 246, 0.3);
    }

    /* Chat Messages */
    .chat-container {
        border-radius: 15px;
        padding: 1em;
        margin: 1em 0;
        background-color: rgba(30, 41, 59, 0.7);
    }

    /* Tool Calls */
    .tool-result {
        background-color: rgba(30, 41, 59, 0.8);
        border-radius: 10px;
        padding: 1em;
        margin: 1em 0;
        border-left: 4px solid #3B82F6;
    }
    
    /* Tool Call Cards */
    .tool-call {
        background-color: rgba(30, 58, 138, 0.3);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 3px solid #60A5FA;
    }
    
    .tool-call-header {
        font-weight: bold;
        font-size: 1.1em;
        margin-bottom: 8px;
        color: #93C5FD;
    }

    /* Status Messages */
    .status-message {
        padding: 1em;
        border-radius: 10px;
        margin: 1em 0;
    }
    
    .success-message {
        background-color: rgba(16, 185, 129, 0.2);
        color: #A7F3D0;
        border-left: 4px solid #10B981;
    }
    
    .error-message {
        background-color: rgba(239, 68, 68, 0.2);
        color: #FCA5A5;
        border-left: 4px solid #EF4444;
    }

    /* Sidebar Styling */
    .sidebar .stButton button {
        background-color: rgba(59, 130, 246, 0.2) !important;
        color: white !important;
    }
    
    /* Platform selector */
    .platform-selector {
        background-color: rgba(30, 41, 59, 0.7);
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    
    /* Question category sections */
    .question-category {
        margin-top: 15px;
        padding-top: 10px;
        border-top: 1px solid rgba(148, 163, 184, 0.3);
    }
    
    /* Chat input styling */
    .stTextInput input {
        border-radius: 20px !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        background-color: rgba(30, 41, 59, 0.7) !important;
        color: white !important;
        padding: 10px 15px !important;
    }
    
    /* Chat comparison section */
    .comparison-title {
        color: #93C5FD;
        font-weight: 600;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    </style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "cdp_agent" not in st.session_state:
    st.session_state["cdp_agent"] = None
if "loaded_urls" not in st.session_state:
    st.session_state.loaded_urls = set()
if "knowledge_base_initialized" not in st.session_state:
    st.session_state.knowledge_base_initialized = False
if "selected_platform" not in st.session_state:
    st.session_state.selected_platform = "All Platforms"


def restart_agent():
    """Reset the agent and clear chat history"""
    logger.debug("---*--- Restarting agent ---*---")
    st.session_state["cdp_agent"] = None
    st.session_state["messages"] = []
    st.session_state.knowledge_base_initialized = False
    st.rerun()


def add_message(role, content, tool_calls=None):
    """Add a message to the chat history"""
    message = {"role": role, "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    st.session_state["messages"].append(message)
    return message


def display_tool_calls(container, tool_calls):
    """Display tool calls in the UI"""
    if not tool_calls:
        return
    
    tool_calls_html = ""
    for i, tool_call in enumerate(tool_calls):
        tool_name = tool_call.get("name", "Unknown Tool")
        tool_input = tool_call.get("input", {})
        tool_output = tool_call.get("output", "No output available")
        
        tool_calls_html += f"""
        <div class="tool-call">
            <div class="tool-call-header">🔍 {tool_name}</div>
            <div><strong>Input:</strong> {tool_input}</div>
            <div><strong>Output:</strong> {tool_output}</div>
        </div>
        """
    
    container.markdown(tool_calls_html, unsafe_allow_html=True)


def export_chat_history():
    """Export chat history to markdown format"""
    md_content = "# CDP Support Assistant Chat History\n\n"
    for msg in st.session_state["messages"]:
        role = msg["role"]
        content = msg["content"]
        
        if role == "user":
            md_content += f"## User\n{content}\n\n"
        elif role == "assistant":
            md_content += f"## CDP Support Assistant\n{content}\n\n"
            
            # Add tool calls if they exist
            if "tool_calls" in msg and msg["tool_calls"]:
                md_content += "### Tool Calls\n"
                for tc in msg["tool_calls"]:
                    md_content += f"- **{tc.get('name', 'Tool')}**\n"
                    md_content += f"  - Input: {tc.get('input', {})}\n"
                    md_content += f"  - Output: {tc.get('output', 'No output')}\n"
                md_content += "\n"
    
    return md_content


def about_widget():
    """Display about information in the sidebar"""
    with st.sidebar.expander("ℹ️ About CDP Support Assistant"):
        st.markdown("""
        ### CDP Support Assistant
        
        This AI-powered assistant provides comprehensive support for Customer Data Platforms (CDPs), helping users with:
        
        - Platform-specific "how-to" questions
        - Feature implementation guidance
        - Integration support
        - Cross-platform comparisons
        - Advanced configuration assistance
        
        The assistant is designed to search and extract relevant information from the official documentation of:
        
        - **Segment**: [docs.segment.com](https://segment.com/docs/?ref=nav)
        - **mParticle**: [docs.mparticle.com](https://docs.mparticle.com/)
        - **Lytics**: [docs.lytics.com](https://docs.lytics.com/)
        - **Zeotap**: [docs.zeotap.com](https://docs.zeotap.com/home/en-us/)
        
        Built with Agno AI Framework and Retrieval Augmented Generation.
        """)


def initialize_agent(debug_mode=False, show_tool_calls=True):
    """Initialize or retrieve the CDP Support Agent"""
    if "cdp_agent" not in st.session_state or st.session_state["cdp_agent"] is None:
        logger.info("---*--- Creating CDP Support Agent ---*---")
        agent = get_cdp_support_agent(
            debug_mode=debug_mode,
            show_tool_calls=show_tool_calls
        )
        st.session_state["cdp_agent"] = agent
    return st.session_state["cdp_agent"]


def main():
    ####################################################################
    # App header
    ####################################################################
    st.markdown("<h1 class='main-title'>CDP Support Assistant</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Your expert guide to Customer Data Platforms - helping you navigate Segment, mParticle, Lytics, and Zeotap</p>",
        unsafe_allow_html=True,
    )
    
    # CDP platform badges
    st.markdown("""
    <div class="cdp-platforms">
        <span class="cdp-pill segment-pill">Segment</span>
        <span class="cdp-pill mparticle-pill">mParticle</span>
        <span class="cdp-pill lytics-pill">Lytics</span>
        <span class="cdp-pill zeotap-pill">Zeotap</span>
    </div>
    """, unsafe_allow_html=True)

    ####################################################################
    # Initialize Agent
    ####################################################################
    cdp_agent: Agent = initialize_agent(debug_mode=False, show_tool_calls=True)

    ####################################################################
    # Platform selector
    ####################################################################
    st.sidebar.markdown("### 🔍 Filter by Platform")
    
    # Platform selector styled as a container
    st.sidebar.markdown('<div class="platform-selector">', unsafe_allow_html=True)
    selected_platform = st.sidebar.radio(
        "Choose a CDP platform:",
        ["All Platforms", "Segment", "mParticle", "Lytics", "Zeotap"],
        index=["All Platforms", "Segment", "mParticle", "Lytics", "Zeotap"].index(st.session_state.selected_platform),
        key="platform_selector"
    )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Update selected platform in session state
    st.session_state.selected_platform = selected_platform

    ####################################################################
    # Sample Questions
    ####################################################################
    st.sidebar.markdown("### ❓ Sample Questions")
    
    # Basic "How-to" questions
    st.sidebar.markdown('<div class="question-category">', unsafe_allow_html=True)
    st.sidebar.markdown("#### Basic Setup & Configuration")
    
    # Filter sample questions based on selected platform
    if selected_platform == "All Platforms" or selected_platform == "Segment":
        if st.sidebar.button("📊 Setup Segment Source", use_container_width=True):
            add_message(
                "user",
                "How do I set up a new source in Segment?"
            )
    
    if selected_platform == "All Platforms" or selected_platform == "mParticle":
        if st.sidebar.button("👤 Create mParticle Profile", use_container_width=True):
            add_message(
                "user",
                "How can I create a user profile in mParticle?"
            )
    
    if selected_platform == "All Platforms" or selected_platform == "Lytics":
        if st.sidebar.button("🎯 Build Lytics Audience", use_container_width=True):
            add_message(
                "user",
                "How do I build an audience segment in Lytics?"
            )
            
    if selected_platform == "All Platforms" or selected_platform == "Zeotap":
        if st.sidebar.button("🔄 Zeotap Integration", use_container_width=True):
            add_message(
                "user",
                "How can I integrate my data with Zeotap?"
            )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Bonus feature: Cross-CDP Comparisons
    if selected_platform == "All Platforms":
        st.sidebar.markdown('<div class="question-category">', unsafe_allow_html=True)
        st.sidebar.markdown('<div class="comparison-title">📊 Cross-CDP Comparisons</div>', unsafe_allow_html=True)
        
        if st.sidebar.button("Compare Audience Creation", use_container_width=True):
            add_message(
                "user",
                "How does Segment's audience creation process compare to Lytics'?"
            )
            
        if st.sidebar.button("Compare Data Integration", use_container_width=True):
            add_message(
                "user",
                "What are the differences between mParticle and Zeotap for data integration?"
            )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced questions section
    st.sidebar.markdown('<div class="question-category">', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="comparison-title">🔧 Advanced Questions</div>', unsafe_allow_html=True)
    
    if selected_platform == "All Platforms" or selected_platform == "Segment":
        if st.sidebar.button("Segment Custom Destinations", use_container_width=True):
            add_message(
                "user",
                "How can I set up custom destinations in Segment for real-time data processing?"
            )
            
    if selected_platform == "All Platforms" or selected_platform == "mParticle":
        if st.sidebar.button("mParticle Identity Resolution", use_container_width=True):
            add_message(
                "user",
                "How does identity resolution work in mParticle and how can I configure it for my use case?"
            )
    st.sidebar.markdown('</div>', unsafe_allow_html=True)

    ####################################################################
    # Utility buttons
    ####################################################################
    st.sidebar.markdown("### 🛠️ Utilities")
    col1, col2 = st.sidebar.columns([1, 1])
    with col1:
        if st.button("🔄 New Chat", use_container_width=True):
            restart_agent()
    with col2:
        if st.download_button(
            "💾 Export Chat",
            export_chat_history(),
            file_name="cdp_chat_history.md",
            mime="text/markdown",
            use_container_width=True,
        ):
            st.sidebar.success("Chat history exported!")

    ####################################################################
    # About section
    ####################################################################
    about_widget()

    ####################################################################
    # Display chat history
    ####################################################################
    for message in st.session_state["messages"]:
        if message["role"] in ["user", "assistant"]:
            _content = message["content"]
            if _content is not None:
                with st.chat_message(message["role"]):
                    # Display tool calls if they exist in the message
                    if "tool_calls" in message and message["tool_calls"]:
                        display_tool_calls(st.empty(), message["tool_calls"])
                    st.markdown(_content)

    ####################################################################
    # Handle user input
    ####################################################################
    if prompt := st.chat_input("Ask me how to do something in a CDP platform..."):
        # If platform is selected, prefix the question to focus on that platform
        if st.session_state.selected_platform != "All Platforms":
            platform_prefix = f"For {st.session_state.selected_platform}: "
            if not prompt.startswith(platform_prefix) and not st.session_state.selected_platform.lower() in prompt.lower():
                prompt = platform_prefix + prompt
                
        user_message = add_message("user", prompt)
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
    ####################################################################
    # Generate response for user message
    ####################################################################
    last_message = (
        st.session_state["messages"][-1] if st.session_state["messages"] else None
    )
    if last_message and last_message.get("role") == "user":
        question = last_message["content"]
        with st.chat_message("assistant"):
            # Create container for tool calls
            tool_calls_container = st.empty()
            resp_container = st.empty()
            with st.spinner("🔍 Searching documentation..."):
                response = ""
                try:
                    # Run the agent and stream the response
                    run_response = cdp_agent.run(question, stream=True)
                    for _resp_chunk in run_response:
                        # Display tool calls if available
                        if hasattr(_resp_chunk, 'tools') and _resp_chunk.tools and len(_resp_chunk.tools) > 0:
                            display_tool_calls(tool_calls_container, _resp_chunk.tools)

                        # Display response
                        if hasattr(_resp_chunk, 'content') and _resp_chunk.content is not None:
                            response += _resp_chunk.content
                            resp_container.markdown(response)

                    # Add the complete response to message history
                    tools = None
                    if hasattr(cdp_agent, 'run_response') and hasattr(cdp_agent.run_response, 'tools'):
                        tools = cdp_agent.run_response.tools
                    add_message("assistant", response, tools)
                except Exception as e:
                    error_message = f"""
                    <div class="error-message">
                        <strong>Error:</strong> I encountered an issue while processing your request: {str(e)}
                        <br/><br/>
                        Please try reformulating your question or selecting a specific CDP platform from the sidebar.
                    </div>
                    """
                    add_message("assistant", error_message)
                    st.markdown(error_message, unsafe_allow_html=True)


if __name__ == "__main__":
    main()