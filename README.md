# **CDP Support Assistant - RAG-based AI Chatbot** 🤖📊

#### **An intelligent assistant for Customer Data Platforms (CDPs), leveraging Retrieval-Augmented Generation (RAG) to provide comprehensive guidance on Segment, mParticle, Lytics, and Zeotap.**

---

## 🚀 **Overview**

CDP Support Assistant is an **AI-powered chatbot** designed to help users navigate the complexities of modern Customer Data Platforms. This application provides detailed answers to "how-to" questions related to four major CDPs:

- **Segment** - Comprehensive setup and integration guidance
- **mParticle** - User profile and data management assistance
- **Lytics** - Audience segmentation and personalization support
- **Zeotap** - Data integration and compliance information

The assistant intelligently extracts information from official documentation to guide users on performing specific tasks within each platform.

🔍 **How it Works:**
1. **User asks a "how-to" question** (e.g., "How do I set up a new source in Segment?")
2. **The system searches official CDP documentation**
3. **Google Gemini AI processes the query**, enhancing responses with retrieval-augmented knowledge
4. **A well-structured, accurate response is generated**, with step-by-step instructions

![CDP Support Assistant Screenshot](https://github.com/user-attachments/assets/7ec9edae-5def-42dd-878c-96a4a7ca1b38)

---

## 🛠 **Tech Stack & Architecture**

### Core Components:
- **Frontend:** [Streamlit](https://streamlit.io/) - Powers the interactive UI
- **AI Model:** [Google Gemini](https://ai.google.dev/) - Handles natural language understanding
- **Vector Database:** [ChromaDB](https://trychroma.com/) - Stores document embeddings
- **RAG Framework:** [LangChain](https://python.langchain.com/) - Orchestrates knowledge retrieval
- **Development Framework:** [Agno AI](https://github.com/agno-ai) - Simplifies agent creation

### Key Data Sources:
- [Segment Documentation](https://segment.com/docs/?ref=nav)
- [mParticle Documentation](https://docs.mparticle.com/)
- [Lytics Documentation](https://docs.lytics.com/)
- [Zeotap Documentation](https://docs.zeotap.com/home/en-us/)

### Technical Architecture:
```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   Streamlit UI   │────▶│  CDP Support     │────▶│ Vector Database  │
│                  │◀────│  Agent           │◀────│ (ChromaDB)       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                               │    ▲
                               │    │
                               ▼    │
                         ┌──────────────────┐
                         │  Google Gemini   │
                         │  LLM             │
                         └──────────────────┘
                               │    ▲
                               │    │
                               ▼    │
                         ┌──────────────────┐
                         │  CDP Knowledge   │
                         │  Sources         │
                         └──────────────────┘
```

---

## 📦 **Installation & Setup**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/yourusername/cdp-support-assistant.git
cd cdp-support-assistant
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Set Environment Variables**
Create a `.env` file in the root directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### **4️⃣ Run the App Locally**
```sh
streamlit run app.py
```
The app will be accessible at **`http://localhost:8501`** 🚀

---

## 🐳 **Run with Docker**

### **1️⃣ Build the Docker Image**
```sh
docker build -t cdp-support-assistant .
```

### **2️⃣ Run the Container**
```sh
docker run -p 8501:8501 --env-file .env cdp-support-assistant
```

Visit **`http://localhost:8501`** in your browser to interact with the assistant.

---

## 🎯 **Key Features**

### Core Functionalities:
✅ **"How-to" Question Answering** - Clear step-by-step guidance for CDP tasks
✅ **Documentation Extraction** - Intelligent retrieval from official documentation
✅ **Question Variation Handling** - Understands different phrasings of the same question

### Bonus Features:
✅ **Cross-CDP Comparisons** - Compare approaches between different platforms
✅ **Advanced Configuration Support** - Guidance on complex platform-specific features
✅ **Platform-Filtering UI** - Focus questions on specific CDPs

### Technical Highlights:
✅ **Retrieval-Augmented Generation** - Combines retrieved knowledge with LLM capabilities
✅ **Document Chunking & Embedding** - Optimized for technical documentation
✅ **Streaming Responses** - Real-time answer generation with tool call visibility

---

## 📖 **Usage Examples**

### Basic "How-to" Questions:
- "How do I set up a new source in Segment?"
- "How can I create a user profile in mParticle?"
- "How do I build an audience segment in Lytics?"
- "How can I integrate my data with Zeotap?"

### Advanced Questions:
- "How do I implement real-time personalization using Segment and Lytics?"
- "What's the best way to handle identity resolution in mParticle?"
- "How can I ensure GDPR compliance when using Zeotap?"

### Cross-CDP Comparisons:
- "How does Segment's audience creation process compare to Lytics'?"
- "What are the differences between mParticle and Zeotap for data integration?"

---

## 📈 **Evaluation Criteria**

This project was developed according to the following evaluation criteria:

1. **Accuracy and Completeness** - Thorough and correct answers to CDP questions
2. **Code Quality** - Well-structured, maintainable codebase following best practices
3. **Question Handling** - Robust handling of variations in phrasing and terminology
4. **Bonus Feature Implementation** - Cross-CDP comparisons and advanced configurations
5. **User Experience** - Intuitive interface with clear information presentation

---

## 📚 **Project Structure**

```
cdp-support-assistant/
├── app.py                  # Main Streamlit application
├── agentic_rag.py          # Core RAG implementationn
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker configuration
└── README.md               # Project documentation
```

---

## 🚧 **Future Enhancements**

🔜 **Enhanced Topic Classification** - Automatically categorize questions by CDP feature
🔜 **Interactive Tutorials** - Guided walkthroughs for common CDP tasks
🔜 **Cross-Documentation Semantic Search** - Find related concepts across platforms
🔜 **UI Customization Options** - Allow users to adjust the interface to their preferences
🔜 **Knowledge Base Expansion** - Add support for additional CDP platforms

---

## 🤝 **Contributing**

Want to improve the CDP Support Assistant?

1. **Fork this repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-improvement`)
3. **Commit your changes** (`git commit -m 'Add amazing improvement'`)
4. **Push to the branch** (`git push origin feature/amazing-improvement`)
5. **Open a Pull Request**

---

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 **Acknowledgements**

- [Agno AI](https://github.com/agno-ai) for the agent development framework
- [Google Gemini](https://ai.google.dev/) for the powerful LLM capabilities
- [Streamlit](https://streamlit.io/) for the interactive web application framework
- The documentation teams at Segment, mParticle, Lytics, and Zeotap
