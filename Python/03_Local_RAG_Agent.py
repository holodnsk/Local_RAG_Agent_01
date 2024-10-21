import os
from loguru import logger
import json
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# Настройка логирования с использованием loguru
logger.add("log/03_Local_RAG_Agent.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

### LLM
logger.debug('LLM')
from langchain_ollama import ChatOllama
local_llm = "llama3.2:3b-instruct-fp16"
llm = ChatOllama(model=local_llm, temperature=0)
llm_json_mode = ChatOllama(model=local_llm, temperature=0, format="json")

def get_index_db():
    logger.debug('...get_index_db')
    # Создание векторных представлений (Embeddings)
    logger.debug('Embeddings')
    from langchain_huggingface import HuggingFaceEmbeddings
    model_id = 'intfloat/multilingual-e5-large'
    model_kwargs = {'device': 'cpu'} # Настройка для использования CPU (можно переключить на GPU)
    # model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_id,
        model_kwargs=model_kwargs
    )

    db_file_name = 'db/db_02'
    # Загрузка векторной Базы-Знаний из файла
    logger.debug('Загрузка векторной Базы-Знаний из файла')
    file_path = db_file_name + "/index.faiss"
    import os.path
    # Проверка наличия файла с векторной Базой-Знаний
    if os.path.exists(file_path):
        logger.debug('Уже существует векторная База-знаний')
        # Загрузка существующей Базы-Знаний
        db = FAISS.load_local(db_file_name, embeddings, allow_dangerous_deserialization=True)

    else:
        logger.debug('Еще не создана векторная База-Знаний')
        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        # Загрузка документов
        logger.debug('Загрузка документов')
        from langchain_community.document_loaders import WebBaseLoader
        docs = [WebBaseLoader(url).load() for url in urls]
        docs_list = [item for sublist in docs for item in sublist]

        # Split documents
        logger.debug('Split documents')
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=200
        )
        doc_splits = text_splitter.split_documents(docs_list)
        # Создание векторной Базы-Знаний из chunks
        logger.debug('Векторная База-Знаний')
        db = FAISS.from_documents(doc_splits, embeddings)

        # Сохранение созданной Базы-Знаний в файл
        logger.debug('Сохранение векторной Базы-Знаний в файл')
        db.save_local(db_file_name)

    return db

def get_router(topic):
    logger.debug('...get_router')
    # Prompt
    router_instructions = """You are an expert at routing a user question to a vectorstore or web search.

    The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.

    Use the vectorstore for questions on these topics. For all else, and especially for current events, use web-search.

    Return JSON with single key, datasource, that is 'websearch' or 'vectorstore' depending on the question."""

    router = llm_json_mode.invoke(
        [SystemMessage(content=router_instructions)]
        + [HumanMessage(content=topic)]
    )
    return router

def get_retrieval_grader(retriever, question):
    logger.debug('...get_retrieval_grader')

    ### Retrieval Grader

    # Doc grader instructions
    doc_grader_instructions = """You are a grader assessing relevance of a retrieved document to a user question.

    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant."""

    # Grader prompt
    doc_grader_prompt = """Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}. 

    This carefully and objectively assess whether the document contains at least some information that is relevant to the question.

    Return JSON with single key, binary_score, that is 'yes' or 'no' score to indicate whether the document contains at least some information that is relevant to the question."""

    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    logger.debug(f'doc_txt = {doc_txt}')
    doc_grader_prompt_formatted = doc_grader_prompt.format(
        document=doc_txt, question=question
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=doc_grader_instructions)]
        + [HumanMessage(content=doc_grader_prompt_formatted)]
    )
    return result

# TODO
### Generate


if __name__ == "__main__":
    # Основной блок программы: инициализация, построение базы и генерация ответа
    db = get_index_db()
    # Create retriever
    logger.debug('Create retriever')
    retriever = db.as_retriever(k=3)

    topic = "Who is favored to win the NFC Championship game in the 2024 season?"
    logger.debug(topic)
    router = get_router(topic)
    logger.debug(json.loads(router.content))

    topic = "What are the models released today for llama3.2?"
    logger.debug(topic)
    router = get_router(topic)
    logger.debug(json.loads(router.content))

    topic = "What are the types of agent memory?"
    logger.debug(topic)
    router = get_router(topic)
    logger.debug(json.loads(router.content))

    # Test
    question = "What is Chain of thought prompting?"
    logger.debug(question)
    retrieval_grader_result = get_retrieval_grader(retriever, question)
    logger.debug(json.loads(retrieval_grader_result.content))