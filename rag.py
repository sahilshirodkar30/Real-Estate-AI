from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import warnings
warnings.filterwarnings("ignore")



load_dotenv()

CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = "real_estate"

llm = None
vector_store = None

def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )

        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )
def process_urls(urls):
    '''
        Scrape data from the supplied urls
        and stores them in a vector db
        :param urls:
        :return:
        '''

    #print("initialization")
    yield "Initialize components"
    initialize_components()

    yield "Resetting vector_store"
    vector_store.reset_collection()


    #print("load data")

    yield "Load data"
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()


    yield "Split text"
    #print("split text")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    #print("add doc to vector DB")
    yield "Add docs to the vector db"

    uuids = [str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)




def generate_answer(query):

    global llm, vector_store

    if not vector_store:
        raise RuntimeError("vectorDB is not initialized")
    # 1) Prompt for context stuffing
    system_tmpl = (
        "Use the provided context to answer the user question.\n"
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [("system", system_tmpl), ("human", "{input}")]
    )

    # 2) Create the combine-documents chain (LLM over stuffed context)
    qa_chain = create_stuff_documents_chain(llm, prompt)

    # 3) Make retriever from the vector store
    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, qa_chain)

    # 4) Run chain and extract answer/sources
    out = rag_chain.invoke({"input": query})
    answer = out["answer"]
    docs = out["context"]  # list[Document]

    # Extract unique non-empty sources
    sources = [d.metadata.get("source", "") for d in docs]
    sources_str = " ".join(sorted(set(s for s in sources if s)))

    return answer, sources_str


if __name__ == "__main__":
    urls = [ "",""
    ]

    process_urls(urls)

    #results = vector_store.similarity_search(
       # "30 year morgate rate",
       # k=2
    #)
    #print(results)
    query = "Give me a brief one paragraph summary of each article link uploaded."
    answer,sources = generate_answer(query)
    print(f"\nSources: {sources}")
