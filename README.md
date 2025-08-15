# Chat with a Financial Agent (10-K Filings Analysis)

This project is an **AI-powered financial assistant** built with [LangChain](https://www.langchain.com/), [Gradio](https://www.gradio.app/), and [Groq](https://groq.com/).  
It can:

- Answer **financial questions** based on 10-K filings for **Google, Microsoft, and NVIDIA**.
- Break complex queries into **sub-questions**.
- Retrieve **accurate data** from a Chroma vector store.
- Politely handle **generic questions** (e.g., greetings, unrelated topics).

---

## Features

- **Financial Q&A**: Ask about revenues, growth, and financial metrics from SEC 10-K filings.
- **Automatic Query Decomposition**: Break down multi-part queries into smaller, answerable sub-queries.
- **RAG-powered Search**: Retrieve answers from embedded documents.
- **Generic Question Handling**: Respond with plain text for out-of-scope queries.

---

## Installation

### Clone the repository

```bash
[git clone https://github.com/your-username/financial-agent.git](https://github.com/amirthabala/financial-agentic-chatbot.git)
cd financial-agentic-chatbot
```

### Install dependencies

Make sure you have Python 3.10+ installed, then run:

```bash
pip install -r requirements.txt
```

### Get a Groq API Key

1. Go to Groq Console. (https://groq.com/)
2. Create an account (if you don’t have one).
3. Navigate to the API Keys section.
4. Generate a new key.
5. Copy the key.

### Environment Setup

Create a .env file in the root of your project and add:

```bash
GROQ_API_KEY=your_groq_api_key_here
```

### Running the App

Once dependencies and environment variables are set:

```bash
python3 main.py
```

This will:

- Load or create a Chroma vector database from your ./documents folder.
- Launch the Gradio interface locally.
- Provide a public share link for remote access.

## Example Usage

### Example 1 — Generic Question

### User:

```bash
Hey!
```

### Bot Response (Plain Text):

```bash
Hey, how's it going?
```

### User:

```bash
Thank You
```

### Bot Response (Plain Text):

```bash
You're welcome.
```

### Example 2 — Simple Direct Financial Query

### User:

```bash
What was Microsoft's total revenue in 2023?
```

### Bot Response (JSON):

```bash
{
    "query": "What was Microsoft's total revenue in 2023?",
    "answer": "$211,922 million",
    "reasoning": "Calculated by subtracting $33.2 billion from the revenue of fiscal year 2024",
    "sub_queries": [],
    "sources": [
        {
            "company": "Microsoft",
            "year": "2024",
            "excerpt": "ITEM 7, Fiscal Year 2024 Compared with Fiscal Year 2023"
        }
    ]
}
```

### Example 3 — Comparative Query

### User:

```bash
How did NVIDIA's data center revenue grow from 2022 to 2023?
```

### Bot Response (JSON):

```bash
{
    "query": "How did NVIDIA's data center revenue grow from 2022 to 2023?",
    "answer": "NVIDIA's data center revenue grew by 41% from 2022 to 2023.",
    "reasoning": "The growth rate was calculated using the revenue for 2022 and 2023, which were obtained from the 10-K filing documents.",
    "sub_queries": [
        "What was NVIDIA's data center revenue in 2022?",
        "What was NVIDIA's data center revenue in 2023?"
    ],
    "sources": [
        {
            "company": "NVIDIA Corporation",
            "year": "2023",
            "excerpt": "ITEM 15 of the SEC 10-K filing document"
        }
    ]
}
```

### Example 4 — Cross-Company Analysis

### User:

```bash
How did each company's operating margin change from 2022 to 2024?
```

### Bot Response (JSON):

```bash
{
    "query": "How did each company's operating margin change from 2022 to 2024?",
    "answer": "The operating margin changes for Google, Microsoft, and Nvidia from 2022 to 2024 are not explicitly stated in the provided documents. However, based on available data, Google's operating margin likely improved due to increased revenues and operating income. Microsoft's operating income increased, suggesting a possible improvement in operating margin, but exact figures are not provided. Nvidia's operating margin calculation is not possible with the given information.",
    "reasoning": "The answer is based on observations from the 'Search from Documents' tool, which analyzed SEC 10-K filings for Alphabet (Google), Microsoft, and Nvidia. The filings provided some financial data but lacked explicit operating margin information for the years in question.",
    "sub_queries": [
        "How did Google's operating margin change from 2022 to 2024?",
        "How did Microsoft's operating margin change from 2022 to 2024?",
        "How did Nvidia's operating margin change from 2022 to 2024?"
    ],
    "sources": [
        {
            "company": "Alphabet (Google)",
            "year": "2022 and 2024",
            "excerpt": "ITEM 15 of the SEC 10-K filing for Alphabet (2024) and ITEM 8 of the SEC 10-K filing for Alphabet (2023)"
        },
        {
            "company": "Microsoft",
            "year": "2022 and 2024",
            "excerpt": "ITEM 7 and ITEM 8 of the Form 10-K for Microsoft Corporation"
        },
        {
            "company": "Nvidia",
            "year": "2022, 2023, and 2024",
            "excerpt": "ITEM 7 of Nvidia's SEC 10-K filing"
        }
    ]
}
```

### Example 5 — Segment Analysis

### User:

```bash
What percentage of Google's revenue came from cloud in 2023?
```

### Bot Response (JSON):

```bash
{
    "query": "What percentage of Google's revenue came from cloud in 2023?",
    "answer": "10.75%",
    "reasoning": "The answer is calculated by dividing Google's cloud revenue in 2023 by Google's total revenue in 2023 and then multiplying by 100.",
    "sub_queries": ["What was Google's total revenue in 2023?", "What was Google's cloud revenue in 2023?"],
    "sources": [
        {
            "company": "Alphabet Inc.",
            "year": "2023",
            "excerpt": "ITEM 8 and ITEM 15 of the SEC 10-K filing document"
        },
        {
            "company": "Alphabet Inc.",
            "year": "2023",
            "excerpt": "ITEM 8 of the SEC 10-K filing document for cloud revenue"
        }
    ]
}
```

### Example 6 — AI Strategy

### User:

```bash
Compare AI investments mentioned by all three companies in their 2024 10-Ks
```

### Bot Response (JSON):

```bash
{
    "query": "Compare AI investments mentioned by all three companies in their 2024 10-Ks",
    "answer": "Google, Microsoft, and Nvidia have all made significant investments in AI, with a focus on developing and acquiring AI platform services, incorporating AI into existing products and services, and launching new AI-powered products and applications.",
    "reasoning": "The comparison is based on the information gathered from the 2024 10-K filings of the three companies, which highlights their commitment to investing in AI and their efforts to innovate and provide helpful products and services.",
    "sub_queries": [
        "What AI investments are mentioned in Google's 2024 10-K?",
        "What AI investments are mentioned in Microsoft's 2024 10-K?",
        "What AI investments are mentioned in Nvidia's 2024 10-K?"
    ],
    "sources": [
        {
            "company": "Alphabet Inc.",
            "year": "2024",
            "excerpt": "ITEM 15, Alphabet Inc., 2024"
        },
        {
            "company": "Microsoft",
            "year": "2024",
            "excerpt": "ITEM 1A, Microsoft, 2024"
        },
        {
            "company": "Nvidia",
            "year": "2024",
            "excerpt": "ITEM 1A, Nvidia, 2024"
        }
    ]
}
```

### Example 7 — Unrelated Questions

### User:

```bash
How to make Dosas?
```

### Bot Response (JSON):

```bash
That's a great question, but I'm afraid it's outside my knowledge domain, which is focused on financial data in 10-K filings. I can't provide you with a recipe or instructions on how to make Dosas. Perhaps you can find a good cooking resource or website that can help you with that?
```

## Project Structure

```bash
├── main.py             # Main application entry point
├── loader.py           # Custom document loader
├── requirements.txt    # Python dependencies
├── .env                # Environment variables (API keys)
├── documents/          # Folder containing 10-K filings
├── chroma_store/       # Persisted Chroma vector database
└── README.md           # Documentation
```

## Tech Stack

- Python 3.10+
- LangChain — Agent orchestration
- Groq API — LLM inference
- Chroma — Vector database
- HuggingFace Sentence Transformer - Embeddings
- Gradio — Web UI
