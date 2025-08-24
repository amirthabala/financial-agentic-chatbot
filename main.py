import os
import gradio as gr
from dotenv import dotenv_values
from loader import load_documents_
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts.prompt import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, create_react_agent

# Load environment variables as a dictionary
env_vars = dotenv_values(".env")

# API Keys
groq_api_key = env_vars.get("GROQ_API_KEY")

print("loaded api keys")

# Model
model = "llama-3.3-70b-versatile"

# Initializing Groq LLM
groq_client = ChatGroq(model=model, api_key=groq_api_key)

print("Initialized LLM")
print("Loading Embedding Model")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chroma_db_path = "./chroma_store"

print("Initialized Embedding Model")

if os.path.exists(chroma_db_path):
    print("Fetching loaded documents")
    vectorstore = Chroma(
        persist_directory="./chroma_store", embedding_function=embedding_model
    )
else:
    documents = load_documents_("./documents")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunked_docs = splitter.split_documents(documents)

    print(f"Original docs: {len(documents)}, After chunking: {len(chunked_docs)}")

    vectorstore = Chroma.from_documents(
        chunked_docs, embedding_model, persist_directory="./chroma_store"
    )
    print("Loaded documents to Vector Store")


def get_sub_queries_from_question(user_query):
    sub_query_template_prompt = """
    You are a helpful assistant that generates multiple sub-questions related to an input question.
    You have access to 10-K filings reports for Google, Microsoft, and NVIDIA.

    Goal:
    - Break the input into the smallest possible independent sub-questions.
    - If the question requires a derived metric (like operating margin, growth %, ratios, or comparisons),
      expand it into sub-queries for the underlying quantities needed to compute it.
        * Example: "operating margin" → ["What was revenue in YEAR?", "What was operating income in YEAR?"]
        * Example: "growth %" → ["What was revenue in YEAR1?", "What was revenue in YEAR2?"]

    Rules:
    - Always output ONLY a valid JSON list of strings.
    - Do NOT include explanations, numbering, or extra text.
    - Each sub-query must be self-contained (include the company and year explicitly).
    - If the input is already atomic, return it as a single-element JSON list.

    Input Question:
    {question}
    """

    prompt = PromptTemplate.from_template(sub_query_template_prompt).format(
        question=user_query
    )

    answer = groq_client.invoke(prompt).content

    return answer


def get_queries_answered(user_query):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.invoke(user_query)

    retrieved_context = "\n\n".join(
        [
            "\n".join(
                [
                    f"Page {r_doc.metadata.get('page_number', 'N/A')} | "
                    f"Source: {r_doc.metadata.get('source', 'N/A')} | "
                    f"Year: {r_doc.metadata.get('year', 'N/A')}\n"
                    f"{r_doc.page_content}"
                ]
            )
            for r_doc in retrieved_docs
        ]
    )

    rag_template_prompt = """
    You are a financial analyst assistant.
    Use the following SEC 10-K filing documents to answer the question comprehensively and accurately.
    If the answer is not explicity given, use your knowledge to retrive necessary context, do calculations and try achieving expected answer.

    Question: {input}

    Context:
    {context}

    Answer in a clear, concise paragraph, along with citing section, company name, page number and year.
    """

    prompt = PromptTemplate.from_template(rag_template_prompt).format(
        input=user_query, context=retrieved_context
    )
    answer = groq_client.invoke(prompt).content
    return answer


def handle_generic_questions(user_query):
    generic_template_prompt = """
        If the question is a greeting, thank you, send-off, or unrelated to the financial data in these 10-K filings,
        Respond politely with a short message stating that it is out of your knowledge.
        If it's greeting, just greet/acknowledge appropriately.

        Question: {input}
    """

    prompt = PromptTemplate.from_template(generic_template_prompt).format(
        input=user_query
    )
    answer = groq_client.invoke(prompt).content

    return answer.strip()


def calculator(expression: str):
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


# Define Tools
tools = [
    Tool(
        name="Get sub-queries from User Query",
        func=get_sub_queries_from_question,
        description="Extracts multiple sub-queries from a user's question by breaking it down into the list of smallest possible independent questions.",
    ),
    Tool(
        name="Search from Documents",
        func=get_queries_answered,
        description="Searches the Stored 10K filings data from Chroma database for the given question and return a synthesized answer.",
    ),
    Tool(
        name="Handle Generic Questions",
        func=handle_generic_questions,
        description="Answers Generic questions like greetings, send offs and out of scope topics",
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for doing math operations (like growth %, differences, ratios) after retrieving numbers.",
    ),
]

print("Defined Tools")

# Prompt Message
template_prompt = """
You are a financial data assistant that answers questions using the available tools.

TOOLS:
{tools}

IMPORTANT INSTRUCTIONS:
- Always think step-by-step before answering.
- Use tools when necessary and clearly indicate which one you are using.
- Provide reasoning and evidence for your answer.
- Only give the final answer after gathering all needed observations.

RULES FOR USING TOOLS:
1. If the query is unrelated to financial data in 10-K filings:
   - ONLY use the 'Handle Generic Questions' tool.
   - Return the Observation from that tool as plain text.
   - Do not continue with further thoughts or actions.
   - Output format:
       Final Answer: <plain text response>

2. If the query is DIRECT (explicit, factual, single-company, single-year, no comparison or inference):
   - Use 'Search from Documents' directly.
   - Return the answer with citations.

3. If the query is INDIRECT (comparisons, growth, percentages, trends, "how did it change", "is it higher", etc.):
   - ALWAYS first use 'Get sub-queries from User Query' to expand the question.
   - Answer each sub-query using 'Search from Documents'.
   - If the sub-queries return quantities needed for a derived metric (e.g., revenue + operating income, two years of revenue, net income + shares), 
     then:
       a) Retrieve all required values from the filings,
       b) Perform the calculation using the 'Calculator' tool,
       c) Use the result to form the final answer.
   - Combine results into a final synthesized answer with reasoning.
   
OUTPUT RULES:
- If the query is unrelated to financial data in 10-K filings, ONLY use the 'Handle Generic Questions' tool.
    - Return the Observation from that tool as plain text.
    - Do not continue with further thoughts or actions.
    - Output format:
        Final Answer: <plain text response>

- If the query is related to financial data:
    - If the query is complex or has mutliple data, use necessary tool to break it down and reason.
    - Use relevant tools ('Get sub-queries from User Query' or 'Search from Documents').
    - Final answer MUST be returned in the following JSON structure:
        {{
            "query": "user input query",
            "answer": "Your Final Answer",
            "reasoning": "Your Decision",
            "sub_queries": List of sub queries if any,
            "sources": List of all sources of subqueries in the below format.
                {{
                    "company": "Which company data you are using",
                    "year": "Which year data you are retrieving",
                    "page": "Which page data of the doc you have used to retrieve the answer",
                    "excerpt": "Exact snippet which you have used to retrieve the answer"
                }}
        }}
- Never mix plain text and JSON in the same final output.

Question: {input}
Thought: Describe your reasoning.
Action: Tool name (choose from [{tool_names}])
Action Input: Data or query for the tool
Observation: Tool's output
... (Repeat Thought/Action/Action Input/Observation as needed)
Decision: Explain how you reached the conclusion
Final Answer: (Follow the OUTPUT RULES above)

Thought: {agent_scratchpad}

"""

prompt = PromptTemplate.from_template(template_prompt)

# Initialize the agent
agent = create_react_agent(llm=groq_client, tools=tools, prompt=prompt)

# Initialize the executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=15,
    handle_parsing_errors=True,
).with_config({"run_name": "Agent"})

print("Initialized Agent")


# Gradio Implementation
def add_message(prompt, messages):
    messages.append(gr.ChatMessage(role="user", content=prompt))
    return messages


async def interact_with_langchain_agent(prompt, messages):
    async for chunk in agent_executor.astream({"input": prompt}):
        if "steps" in chunk:
            for step in chunk["steps"]:
                messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=step.action.log,
                        metadata={"title": f"🛠️ Used tool {step.action.tool}"},
                    )
                )
                yield messages
        if "output" in chunk:
            messages.append(gr.ChatMessage(role="assistant", content=chunk["output"]))
            yield messages


with gr.Blocks(fill_height=True) as demo:
    gr.Markdown("# Chat with a Financial Agent")
    chatbot = gr.Chatbot(
        type="messages",
        label="Agent",
        scale=1,
    )
    input = gr.Textbox(
        placeholder="Chat Message",
        interactive=True,
        show_label=False,
        submit_btn=True,
        autofocus=True,
    )

    # Appending user message to chat history and making the input box disabled
    user_msg = input.submit(add_message, [input, chatbot], [chatbot]).then(
        lambda: gr.Textbox(interactive=False), None, [input]
    )

    # Streaming and appending the Bot message to chat history
    bot_msg = user_msg.then(interact_with_langchain_agent, [input, chatbot], chatbot)

    # Enabling back the input box and and resetting the value
    bot_msg.then(
        lambda: gr.Textbox(value="", interactive=True, autofocus=True), None, [input]
    )

demo.launch(debug=True, share=True)
