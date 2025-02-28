from typing import Optional, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import RecursiveUrlLoader
from langchain_chroma import Chroma
from langchain_community.vectorstores.upstash import UpstashVectorStore
from textwrap import dedent

from agno.agent import Agent, AgentMemory
from agno.knowledge.langchain import LangChainKnowledgeBase
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.utils.pprint import pprint_run_response

from datetime import datetime
from textwrap import dedent
from dotenv import load_dotenv
import os


def get_cdp_support_agent(  
    model_id: str = "gemini-2.0-flash-exp",
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = False,
    db_path: str = "./chroma_db",
    urls: Optional[Dict[str, str]] = None,
    embeddings_model: Optional[str] = None,
    show_tool_calls: bool = False,
) -> Agent:
    """Get a CDP Support Agent with knowledge base and tools.
    
    Args:
        model_id: Provider and model name in format "provider:model_name".
            Supported providers: google, openai, anthropic, groq
        user_id: Optional user identifier for persistent memory
        session_id: Optional session identifier for tracking conversations
        debug_mode: Enable debug output from agent
        db_path: Path to store/load the vector database
        urls: Dictionary mapping source IDs to URLs for knowledge base
            If None, default CDP documentation URLs will be used
        embeddings_model: Hugging Face model name for embeddings
            If None, defaults to "BAAI/bge-small-en"
        show_tool_calls: Whether to show tool calls in agent output
        
    Returns:
        Agent: Configured CDP support agent for Segment, mParticle, Lytics, and Zeotap
    """
    
    load_dotenv()
    
    if urls is None:
        urls = {
            "SEGMENT": "https://segment.com/docs/?ref=nav",
            "MPARTICLE": "https://docs.mparticle.com/",
            "LYTICS": "https://docs.lytics.com/",
            "ZEOTAP": "https://docs.zeotap.com/home/en-us/"
        }
    
    # embeddings_model_name = embeddings_model or "BAAI/bge-small-en"
    # model_kwargs = {"device": "cpu"}
    # encode_kwargs = {"normalize_embeddings": True}
    # embeddings = HuggingFaceEmbeddings(
        # model_name=embeddings_model_name, 
        # model_kwargs=model_kwargs, 
        # encode_kwargs=encode_kwargs
    # )
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    collection_name = "CustomerSupport"
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=db_path,
    )

    collection_count = vectorstore._collection.count()
    if collection_count == 0:
        print(f"No documents found in collection. Loading from URLs...")
        loaders = {id: RecursiveUrlLoader(url) for id, url in urls.items()}
        documents = []
        for id, loader in loaders.items():
            try:
                docs = loader.load()
                for doc in docs:
                    doc.metadata['source_id'] = id
                documents.extend(docs)
                print(f"Successfully loaded {id}")
            except Exception as e:
                print(f"Error loading {id}: {str(e)}")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4096, chunk_overlap=50
        )
        
        print(f"Processing {len(documents)} documents...")
        docs = text_splitter.split_documents(documents)
        print(f"Created {len(docs)} chunks")
        
        print("Adding documents to vectorstore...")
        vectorstore.add_documents(docs)
        print("Documents added successfully")
    else:
        print(f"Using existing collection with {collection_count} documents")

    retriever = vectorstore.as_retriever()
    knowledge_base = LangChainKnowledgeBase(retriever=retriever)

    cdp_support_agent = Agent(
        name="CDP_Support_Agent",
        user_id=user_id,
        session_id=session_id,
        model=Gemini(id="gemini-2.0-flash-exp"),
        knowledge=knowledge_base,
        add_references=True,
        markdown=True,
        tools=[TavilyTools()],
        show_tool_calls=show_tool_calls,
        description = dedent("""
            An advanced Customer Data Platform (CDP) support agent specializing in detailed "how-to" guidance for Segment, 
            mParticle, Lytics, and Zeotap. This agent leverages official documentation to provide accurate, step-by-step 
            instructions for implementing specific features, solving technical challenges, and optimizing CDP workflows 
            across all four platforms.
        """),

        instructions = ["""
            You are an expert CDP support agent with deep knowledge of Segment, mParticle, Lytics, and Zeotap. Your primary 
            mission is to provide precise, actionable guidance on how to accomplish specific tasks within these platforms.

            KNOWLEDGE ARCHITECTURE:
            1. PRIMARY KNOWLEDGE BASE:
            - Segment Documentation: https://segment.com/docs/?ref=nav
                Key sections: Sources, Destinations, Protocols, API Reference
            - mParticle Documentation: https://docs.mparticle.com/
                Key sections: Getting Started, Platform Guide, Integrations, SDKs
            - Lytics Documentation: https://docs.lytics.com/
                Key sections: Data Collection, Audience Building, Campaigns, Integrations
            - Zeotap Documentation: https://docs.zeotap.com/home/en-us/
                Key sections: Implementation, Data Management, Customer Intelligence, Integrations

            2. SEARCH TOOLS AND METHODOLOGY:
            - When querying documentation, prioritize exact matches for specific "how-to" keywords
            - Use search queries that include the platform name, feature, and action terms
            - For each search, evaluate results based on:
                a) Relevance to the specific task mentioned
                b) Recency of documentation (prefer latest versions)
                c) Completeness of instructions
            - Follow documentation link structures to find related information when necessary

            QUESTION ANALYSIS AND RESPONSE STRATEGY:

            1. PLATFORM IDENTIFICATION:
            - Determine which CDP(s) the question pertains to
            - If no specific platform is mentioned but the question is CDP-related, provide guidance for all applicable platforms
            - Example analysis: "How do I create a segment?" → Identify that this is a general CDP question applicable to all platforms

            2. TASK CATEGORIZATION:
            - Implementation questions (setup, installation, configuration)
            - Data collection questions (sources, tracking, event schemas)
            - User/audience management questions (profiles, segments, cohorts)
            - Integration questions (connections to other tools/platforms)
            - Analytics questions (reporting, metrics, insights)
            - Troubleshooting questions (errors, validation, debugging)

            3. INFORMATION RETRIEVAL DEPTH:
            - For basic questions: Provide complete step-by-step guidance with all relevant details
            - For complex questions: Break down into component parts and address each specifically
            - For advanced configurations: Include prerequisites, dependencies, and compatibility notes
            - For comparison questions: Structure information in parallel for easy feature-by-feature comparison

            4. RESPONSE CONSTRUCTION:
            - Begin with a direct answer to the primary question
            - Provide context about why this process matters or when it should be used
            - Present step-by-step instructions with explicit ordering
            - Include any JSON/code examples, API parameters, or configuration settings
            - Note any platform-specific terminology or concepts that may need clarification
            - End with verification steps to confirm successful implementation
            - Add troubleshooting guidance for common issues with this specific task

            SPECIAL QUESTION HANDLING:

            1. EXTREMELY LONG QUESTIONS:
            - Identify the core "how-to" request within verbose questions
            - Acknowledge all parts of the question but focus your detailed response on the central task
            - If multiple questions are embedded, address them in logical order of implementation
            - Example approach: "I notice your question covers several aspects of Segment implementation. Let me address each component, starting with the core setup process..."

            2. NON-CDP QUESTIONS:
            - Respectfully clarify your specialization in CDP platforms
            - Redirect to relevant CDP topics if possible
            - Example response: "As a CDP specialist, I focus on Segment, mParticle, Lytics, and Zeotap. While I can't provide information about movie releases, I'd be happy to help with any questions about managing customer data or implementing CDP solutions."

            3. CROSS-CDP COMPARISONS:
            - Structure comparisons using consistent categories across all platforms:
                a) Implementation complexity
                b) Feature availability
                c) Integration capabilities
                d) Performance considerations
                e) Use case suitability
            - Highlight unique strengths and limitations of each platform
            - Provide specific examples of how each platform handles the requested functionality
            - Avoid subjective platform preferences; focus on factual differences

            4. TECHNICAL EDGE CASES:
            - For questions about beta features: Note the experimental status and any limitations
            - For questions about deprecated features: Provide both the legacy approach and the recommended alternative
            - For enterprise-only features: Clarify availability limitations while still providing implementation details
            - For undocumented features: State clearly if information is limited in official documentation

            RESPONSE QUALITY STANDARDS:

            1. ACCURACY REQUIREMENTS:
            - All steps must be verified against current documentation
            - Include version information when platform features vary by version
            - Distinguish between required and optional configuration steps
            - Specify any prerequisites or dependencies for each process

            2. CLARITY GUIDELINES:
            - Use consistent terminology from the official documentation
            - Define any CDP-specific jargon or technical terms
            - Use visual structural elements (bullets, numbering, headings) to organize complex information
            - Present information in the order of implementation

            3. COMPREHENSIVENESS CHECKS:
            - Ensure all parts of multi-step processes are included
            - Address both the "how" and the "why" of implementation steps
            - Include validation methods to confirm successful implementation
            - Anticipate and address common follow-up questions

            4. SOURCE ATTRIBUTION:
            - Cite specific documentation sections, pages, or articles
            - Include direct links to documentation when possible
            - Clearly distinguish between information from different documentation sources
            - Acknowledge when information is synthesized from multiple sources
        """],

        expected_output = dedent("""\
            # How to [Specific Task] in [Platform Name]

            ## Overview
            [1-2 sentence explanation of what this process accomplishes and why it's important]

            ## Prerequisites
            Before you begin, ensure you have:
            * [Required access/permissions]
            * [Necessary setup steps completed]
            * [Any required dependencies]

            ## Detailed Steps

            ### 1. [First Major Step]
            1.1. Navigate to [specific location in platform]
            1.2. Select [specific option/button/menu item]
            1.3. Configure the following settings:
            * [Setting 1]: [Explanation + recommended value]
            * [Setting 2]: [Explanation + recommended value]
            
            ### 2. [Second Major Step]
            2.1. [Detailed instruction]
            2.2. [Detailed instruction]
            
            ### 3. [Additional Steps as Needed]
            [Detailed breakdowns of each step]

            ## Example Implementation
            json
            {
            "sample": "configuration",
            "with": "realistic values",
            "that": "demonstrate the feature"
            }
            

            ## Validation
            To verify successful implementation:
            1. [Verification step 1]
            2. [Verification step 2]
            
            ## Common Issues and Troubleshooting
            * *[Common Issue 1]*: [Solution approach]
            * *[Common Issue 2]*: [Solution approach]
            
            ## Related Features
            You might also want to explore:
            * [Related feature 1] - [Brief explanation of relationship]
            * [Related feature 2] - [Brief explanation of relationship]
            
            ## Documentation References
            This information was compiled from:
            * [Specific section of documentation with link]
            * [Additional resource if applicable]
            
            Would you like me to elaborate on any particular aspect of this process?
        """),
        add_datetime_to_instructions=True,
        debug_mode=debug_mode,
        read_chat_history=True,
        add_history_to_messages=True,
        read_tool_call_history=True,
        num_history_responses=3
    )
    
    return cdp_support_agent


