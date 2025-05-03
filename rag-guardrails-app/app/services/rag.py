from sentence_transformers import SentenceTransformer
import faiss
import json
from langfuse import Langfuse
from langfuse.decorators import observe, langfuse_context
from litellm import completion
import litellm
import os
from nemoguardrails import LLMRails, RailsConfig
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
# from models.ChatOpenRouter import ChatOpenRouter
from app.models.ChatOpenRouter import ChatOpenRouter

from nemoguardrails.llm.providers import register_llm_provider
import nest_asyncio

load_dotenv()

# Initialize Langfuse client
langfuse = Langfuse()
litellm.success_callback = ["langfuse"] # log input/output to lunary, mlflow, langfuse, helicone
config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yml")
# config_path = "/Users/piyushkumarjain/Projects/github/genai/rag-guardrails-app/app/config/config.yml"
print(f"Loading config from: {config_path}")
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found at {config_path}")

config = RailsConfig.from_path(config_path)# Configure LiteLLM

# Load FAISS index and documents
model = SentenceTransformer('all-MiniLM-L6-v2')


# openrouter_model = ChatOpenRouter(
#     model_name="openrouter/qwen/qwen3-30b-a3b:free",
# )

openrouter_provider = ChatOpenRouter(
    model=os.environ.get("OPENROUTER_MODEL"),
    api_key=os.environ.get("OPENROUTER_API_KEY"),
)

register_llm_provider("custom_llm", ChatOpenRouter)

def get_faiss_index_and_documents(index_path="data/vector_index.faiss", documents_path="data/documents.json"):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not os.path.exists(documents_path):
        raise FileNotFoundError(f"Documents file not found at {documents_path}")
    index = faiss.read_index(index_path)
    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    if not isinstance(documents, list):
        raise ValueError("Documents JSON must be a list")
    return index, documents

@observe(as_type="generation")
async def rag_pipeline(query, index_path="data/vector_index.faiss", documents_path="data/documents.json"):
    index, documents = get_faiss_index_and_documents(index_path, documents_path)
    query_vector = model.encode([query])
    _, indices = index.search(query_vector, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant. Answer the question based on the provided context."
    },
    {
        "role": "user",
        "content": f"Context: {retrieved_docs}\n\nQuestion: {query}"
    }
]
    langfuse_context.update_current_observation(
        input=messages,
        model = os.environ.get("OPENROUTER_MODEL")
)

    try:
        app = LLMRails(config,llm=openrouter_provider, verbose=False)
        response = await app.generate_async(
                                 messages=[{"role": "user", "content": f"{query}"}])
        print(response)
        answer = response
        return answer
    except Exception as e:
        raise e

