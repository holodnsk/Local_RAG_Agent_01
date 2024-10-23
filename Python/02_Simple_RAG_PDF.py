""" Описание модуля
Этот модуль реализует метод генерации ответа на заданную тему, используя OpenAI API.
Основные шаги включают загрузку и обработку PDF-документов, создание векторной Базы-Знаний для поиска по схожести содержимого
и использование OpenAI для генерации ответа.
Векторная База-Знаний хранится и загружается с локального диска для ускорения работы.
"""

import os
from loguru import logger
from langchain_community.vectorstores import FAISS
import openai
from langchain.prompts import ChatPromptTemplate

# Настройка логирования с использованием loguru
logger.add("log/02_Simple_RAG_PDF.log", format="{time} {level} {message}", level="DEBUG", rotation="100 KB", compression="zip")

# Установка API ключа OpenAI
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
os.environ["OPENAI_API_URL"] = "url"

def get_index_db():
    """
    Функция для получения или создания векторной Базы-Знаний.
    Если база уже существует, она загружается из файла,
    иначе происходит чтение PDF-документов и создание новой базы.
    """
    logger.debug('...get_index_db')
    # Создание векторных представлений (Embeddings)
    logger.debug('Embeddings')
    embeddings = FAISS.load_local("db/db_01", "text-embedding-ada-002")

    db_file_name = 'db/db_01'
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
        # Если базы нет, происходит создание новой путем чтения PDF-документов
        # Document loaders
        ## Document loaders: https://python.langchain.com/docs/integrations/document_loaders
        ## PyPDFLoader: https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf
        from langchain_community.document_loaders import PyPDFLoader

        logger.debug(f'Document loaders. dir={dir}')
        dir = 'pdf'
        documents = []
        # Чтение всех PDF-файлов в указанной директории
        for root, dirs, files in os.walk(dir):
            for file in files:
                if file.endswith(".pdf"):
                    logger.debug(f'root={root} file={file}')
                    loader = PyPDFLoader(os.path.join(root, file))
                    documents.extend(loader.load())

        # Разделение документов на меньшие части (chunks)
        logger.debug('Разделение на chunks')
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
        source_chunks = text_splitter.split_documents(documents)
        logger.debug(type(source_chunks))
        logger.debug(len(source_chunks))
        logger.debug(source_chunks[100].metadata)
        logger.debug(source_chunks[100].page_content)

        # Создание векторной Базы-Знаний из chunks
        logger.debug('Векторная База-Знаний')
        db = FAISS.from_documents(source_chunks, embeddings)

        # Сохранение созданной Базы-Знаний в файл
        logger.debug('Сохранение векторной Базы-Знаний в файл')
        db.save_local(db_file_name)

    return db

def get_message_content(topic, index_db, NUMBER_RELEVANT_CHUNKS):
    """
    Функция для извлечения релевантных кусочков текста из Базы-Знаний.
    Выполняется поиск по схожести, извлекаются top-N релевантных частей.
    """
    import re
    logger.debug('...get_message_content: Similarity search')
    docs = db.similarity_search(topic, k=NUMBER_RELEVANT_CHUNKS)
    message_content = re.sub(r'\n{2}', ' ', '\n '.join([f'\n#### {i+1} Relevant chunk ####\n' + str(doc.metadata) + '\n' + doc.page_content + '\n' for i, doc in enumerate(docs)]))
    logger.debug(message_content)
    return message_content

def get_model_response(topic, message_content):
    """
    Функция для генерации ответа модели на основе переданного контекста и вопроса.
    Используется OpenAI для создания ответа.
    """
    logger.debug('...get_model_response')

    # Инициализация модели OpenAI
    logger.debug('OpenAI LLM')
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # или "gpt-4" для более мощной модели
        temperature=0
    )

    # Создание промпта
    prompt = ChatPromptTemplate.from_template("""Ты являешься помощником для выполнения заданий по ответам на вопросы. 
    Вот контекст, который нужно использовать для ответа на вопрос:
    {context} 
    Внимательно подумайте над приведенным контекстом. 
    Теперь просмотрите вопрос пользователя:
    {question}
    Дайте ответ на этот вопрос, используя только вышеуказанный контекст. 
    Используйте не более трех предложений и будьте лаконичны в ответе.
    Ответ:""")

    # Формирование и отправка запроса
    messages = prompt.format_messages(context=message_content, question=topic)
    response = llm.invoke(messages)
    model_response = response.content
    logger.debug(model_response)
    return model_response

if __name__ == "__main__":
    db = get_index_db()
    NUMBER_RELEVANT_CHUNKS = 3
    topic = 'Какие способы интеграции следует использовать'
    logger.debug(topic)
    message_content = get_message_content(topic, db, NUMBER_RELEVANT_CHUNKS)
    model_response = get_model_response(topic, message_content)
