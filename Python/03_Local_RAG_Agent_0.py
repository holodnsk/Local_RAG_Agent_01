import os
from loguru import logger
import json
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage

# Настройка логирования с использованием loguru
logger.add("log/03_Local_RAG_Agent_0.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

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

def get_generated_response(retriever, question):
    logger.debug('...get_generated_response')
    ### Generate
    # Prompt
    rag_prompt = """You are an assistant for question-answering tasks. 

    Here is the context to use to answer the question:

    {context} 

    Think carefully about the above context. 

    Now, review the user question:

    {question}

    Provide an answer to this questions using only the above context. 

    Use three sentences maximum and keep the answer concise.

    Answer:"""

    docs = retriever.invoke(question)
    docs_txt = format_docs(docs)
    rag_prompt_formatted = rag_prompt.format(context=docs_txt, question=question)
    generation = llm.invoke([HumanMessage(content=rag_prompt_formatted)])
    return generation, docs_txt


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_hallucination_grader(generation_txt, docs_txt):
    logger.debug('...get_hallucination_grader')
    ### Hallucination Grader

    # Hallucination grader instructions
    hallucination_grader_instructions = """

    You are a teacher grading a quiz. 

    You will be given FACTS and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) Ensure the STUDENT ANSWER is grounded in the FACTS. 

    (2) Ensure the STUDENT ANSWER does not contain "hallucinated" information outside the scope of the FACTS.

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    hallucination_grader_prompt = """FACTS: \n\n {documents} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER is grounded in the FACTS. And a key, explanation, that contains an explanation of the score."""

    # Test using documents and generation from above
    hallucination_grader_prompt_formatted = hallucination_grader_prompt.format(
        documents=docs_txt, generation=generation_txt
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=hallucination_grader_instructions)]
        + [HumanMessage(content=hallucination_grader_prompt_formatted)]
    )
    return result

def get_answer_grader(question, answer):
    logger.debug('...get_answer_grader')
    ### Answer Grader

    # Answer grader instructions
    answer_grader_instructions = """You are a teacher grading a quiz. 

    You will be given a QUESTION and a STUDENT ANSWER. 

    Here is the grade criteria to follow:

    (1) The STUDENT ANSWER helps to answer the QUESTION

    Score:

    A score of yes means that the student's answer meets all of the criteria. This is the highest (best) score. 

    The student can receive a score of yes if the answer contains extra information that is not explicitly asked for in the question.

    A score of no means that the student's answer does not meet all of the criteria. This is the lowest possible score you can give.

    Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct. 

    Avoid simply stating the correct answer at the outset."""

    # Grader prompt
    answer_grader_prompt = """QUESTION: \n\n {question} \n\n STUDENT ANSWER: {generation}. 

    Return JSON with two two keys, binary_score is 'yes' or 'no' score to indicate whether the STUDENT ANSWER meets the criteria. And a key, explanation, that contains an explanation of the score."""


    # Test using question and generation from above
    answer_grader_prompt_formatted = answer_grader_prompt.format(
        question=question, generation=answer
    )
    result = llm_json_mode.invoke(
        [SystemMessage(content=answer_grader_instructions)]
        + [HumanMessage(content=answer_grader_prompt_formatted)]
    )
    return result

def get_web_search_tool():
    ### Web Search Tool
    from langchain_community.tools.tavily_search import TavilySearchResults
    web_search_tool = TavilySearchResults(k=3)
    return web_search_tool



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

    generation, docs_txt = get_generated_response(retriever, question)
    logger.debug(generation.content)

    result = get_hallucination_grader(generation.content, docs_txt)
    logger.debug(json.loads(result.content))

    # Test
    question = "What are the vision models released today as part of Llama 3.2?"
    answer = "The Llama 3.2 models released today include two vision models: Llama 3.2 11B Vision Instruct and Llama 3.2 90B Vision Instruct, which are available on Azure AI Model Catalog via managed compute. These models are part of Meta's first foray into multimodal AI and rival closed models like Anthropic's Claude 3 Haiku and OpenAI's GPT-4o mini in visual reasoning. They replace the older text-only Llama 3.1 models."

    result = get_answer_grader(question, answer)
    logger.debug(json.loads(result.content))