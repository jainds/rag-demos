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

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
INDEX_PATH = os.path.join(DATA_DIR, 'vector_index.faiss')
DOCS_PATH = os.path.join(DATA_DIR, 'documents.json')

def validate_and_repair_documents(documents_path, default_documents):
    """
    Check if documents.json is valid (non-empty list of dicts with 'text' field). If not, overwrite with default_documents.
    """
    import json
    import os
    try:
        if not os.path.exists(documents_path):
            raise FileNotFoundError
        with open(documents_path, "r", encoding="utf-8") as f:
            docs = json.load(f)
        if not isinstance(docs, list) or len(docs) == 0 or not all(isinstance(d, dict) and 'text' in d for d in docs):
            raise ValueError
    except (FileNotFoundError, json.JSONDecodeError, ValueError):
        # Overwrite with default
        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump(default_documents, f, ensure_ascii=False, indent=2)
        print(f"[Repair] Overwrote {documents_path} with default documents.")

def get_faiss_index_and_documents(index_path=INDEX_PATH, documents_path=DOCS_PATH):
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    # Validate and repair documents.json if needed
    default_documents = [
        {"id": 1, "text": "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."},
        {"id": 2, "text": "The tower was designed by the French civil engineer Gustave Eiffel."},
        {"id": 3, "text": "It was constructed from 1887 to 1889 as the centerpiece of the 1889 World's Fair."}
    ]
    validate_and_repair_documents(documents_path, default_documents)
    with open(documents_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    if not isinstance(documents, list) or len(documents) == 0:
        raise ValueError("Documents JSON must be a non-empty list")
    index = faiss.read_index(index_path)
    return index, documents

@observe(as_type="generation")
async def rag_pipeline(query, index_path=INDEX_PATH, documents_path=DOCS_PATH):
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

