from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

loaader = TextLoader('/Users/bishikachhetri/Desktop/chatbot/ALLINONE.TXT')
documents = loaader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap = 100,
    separators= ['\n\n', '\n',',' ' ','']
)
splitted_text = splitter.split_documents(documents)

vectorstore = Chroma(
    collection_name='single.vdb',
    embedding_function= embeddings,
    persist_directory='all_in_one_insurance.db'
)
vectorstore.add_documents(splitted_text)


