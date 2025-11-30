from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from src.settings import get_settings

settings = get_settings()
loader = TextLoader("src/facts/facts.txt")
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=200, 
    chunk_overlap=0
  )
docs = loader.load_and_split(splitter)
embeddings = OpenAIEmbeddings(
    openai_api_key=settings.openai_api_key
)

emb = embeddings.embed_query("Hello world")
print(emb)
