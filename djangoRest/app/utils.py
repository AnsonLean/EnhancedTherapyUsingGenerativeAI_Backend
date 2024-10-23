from django.conf import settings

#For my functions
import os
from langchain.chains import RetrievalQA
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

#Instantiate the API key
os.environ['OPENAI_APIKEY'] = "your api key"

#Line 14-25 is to read the csv file and store the data in a dictionary
topic_dict = {}
csv_file = open("grouped_df_tableau.csv", "r")

for index, line in enumerate(csv_file.readlines()):
    if index == 0:
        continue

    line = line.strip()
    values = line.split(",")
    topic_dict[values[0]] = (int(values[1]), float(values[2]))

csv_file.close()

# Load existing vector store
persist_directory = "UserKnowledgeBase"
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_APIKEY'])

#Function to load existing vector db
def load_existing_vectorstore(persist_directory, embeddings):
    """
    Function to load the existing vector store

    Args:
    - persist_directory: The directory where the vector store is saved
    - embeddings: The embedding function to use for the vector store

    Returns:
    - vectordb: The loaded vector store
    """
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

# Function to create the QA chain
def create_qa_chain(vectordb, embeddings):
    """
    Function to create the QA chain

    Args:
    - vectordb: The vector store to use for the QA chain
    - embeddings: The embedding function to use for the QA chain

    Returns:
    - qa_chain: The QA chain object

    """
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key= os.environ['OPENAI_APIKEY'], model="gpt-4o-2024-05-13"),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

#Function to create the QA chain with a prompt
def create_qa_chain_cluster(vectordb, embeddings, PROMPT):
    """
    Function to create the QA chain with a prompt

    Args:
    - vectordb: The vector store to use for the QA chain
    - embeddings: The embedding function to use for the QA chain
    - PROMPT: The prompt to use for the QA chain

    Returns:
    - qa_chain: The QA chain object

    """
    retriever = vectordb.as_retriever()
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key= os.environ['OPENAI_APIKEY'], model="gpt-4o-2024-05-13"),
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={
            "prompt": PROMPT
        },
        return_source_documents=True,
    )


# Function to process the LLM response and print sources
def process_llm_response(llm_response):
    """
    Function to process the LLM response and print sources

    Args:
    - llm_response: The response from the LLM model

    Returns:
    - response: The processed response
    """

    response = llm_response['result']

    return response

def send_query_to_api(query):
    """
    Function to send query to api

    Args:
    - query: The query to send to the API

    Returns:
    - response: The response from the LLM
    """
    #Load existing vector db
    vectordb = load_existing_vectorstore(persist_directory, embeddings)
    
    #Update the qa_chain
    qa_chain = create_qa_chain(vectordb, embeddings)
    llm_response = qa_chain.invoke({"query": query})
    response = process_llm_response(llm_response)
    return response

from langchain_core.prompts import PromptTemplate

#Function to cluster topic
def send_text_to_api(text, session_id):
    """
    Function to cluster topic 

    Args:
    - text: The text to send to the API
    - session_id: The session ID of the text

    Returns:
    - True: If the text is successfully sent to the API
    """

    prompt_template = """
    You are reading documents storing about the transcripts of different session. 
    You would like to know more about the topics or information in one of the transcripts{context}. 
    You first have to check the few lines of the documents to select the document as queried by the user. For example, you can ask the question like: What is the topic in TS0005 session by Billy? and you have to look for the document that has "Therapy Session ID: TS0003" and "Billy"
    The topics refers to a summary of any information obtained during the conversation in two to six words.
    Just give the topic and do not include any other information.
    Just give the output in all lowercase

    Choose the topic from this the list below, and if the topic is not in this list, assign it to a new topic:
    topics = [
        "anxiety management",
        "asthma management",
        "avoiding doi",
        "being assertive with flatmate about moving out",
        "better oral health",
        "birth control",
        "changing approach to disease",
        "charging battery",
        "completion of community service",
        "compliance with rules",
        "diabetes management",
        "diagnosis",
        "increasing self-confidence",
        "managing life",
        "opening up",
        "overcoming issues at school",
        "problem recognition",
        "providing information on medicines",
        "recognising success",
        "reducing alcohol consumption",
        "reducing drug use",
        "reducing gambling",
        "reducing recidivism",
        "reducing self-harm",
        "smoking cessation",
        "supporting client to live in more alignment with her values",
        "taking medicine / following medical procedure",
        "taking steps towards getting help with day-care",
        "unidentifiable",
        "weight loss",
        "relationship issues",
    ]

    EXAMPLE #1:
    What brings you here today?
    I have some issue with drinking alcohol.

    question: What is the topic of this transcript? 
    drinking alochol

    EXAMPLE #2:
    I know that you have been struggling with weight loss.

    question: What is the topic of this transcript?
    weight loss

    EXAMPLE #3:
    So, let's see. It looks that you put-- You drink alcohol at least four times a week on average-
    Mm-hmm.
    and you usually have three to four drinks when you do drink.
    Usually three drinks and glasses of wine.
    Okay. That's at least 12 drinks a week.
    Something like that.
    Okay. Just so you know, my role, um, when we talk about alcohol use, is just to share information about risk and to help patients who want help. This is different than telling them what I think they should do. I don't do that.
    Okay.
    Uh, what else can you tell me about your drinking.

    question: What is the topic of this transcript?
    alcohol use

    Question: {question}
    """

    PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"],
                )

    # Append the specific query to ask for the topic
    query = f"What is the topic discussed in this session (Session ID: {session_id})?\n{text}"

    # Load existing vector db
    vectordb = load_existing_vectorstore(persist_directory, embeddings)

    # Update the qa_chain
    qa_chain = create_qa_chain_cluster(vectordb, embeddings, PROMPT)

    # Send the query to the API
    llm_response = qa_chain.invoke({"query": query})
    topic = process_llm_response(llm_response)

    try:
        #Get length of session from text (line 5)
        session_length_text = text.split("\n")[4]
        session_length = float(session_length_text.split(" ")[3].strip())
    except:
        return False

    # Update the topic_dict with the session_id
    if topic not in topic_dict:
        topic_dict[topic] = (1, session_length)
    else:
        freq, avg_length = topic_dict[topic]
        avg_length = round((avg_length*freq + session_length) / (freq + 1),2)
        freq = freq + 1
        topic_dict[topic] = (freq, avg_length)

    return True

"""Add documents to vector db"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# Define the directories and file pattern
NEW_DOCUMENTS_DIR = r"new_data"
OLD_DOCUMENTS_DIR = r"data"
FILE_PATTERN = "./*.txt"

def add_document_to_db(input):
    """
    Function to add documents to the vector database

    Args:
    - input: The input to the function

    Returns:
    - message: The message indicating the number of documents added or the error message
    """

    # Check if the new_data directory is empty
    if not os.listdir(NEW_DOCUMENTS_DIR):
        return "Directory is empty. Nothing is added."
    
    try:
    
        #Load existing vector db
        vectordb = load_existing_vectorstore(persist_directory, embeddings)

        # Load new documents
        loader = DirectoryLoader(
            NEW_DOCUMENTS_DIR,
            glob=FILE_PATTERN,
            loader_cls=lambda p: TextLoader(p, encoding="utf-8")
        )
        documents = loader.load()

        # Move the files to the old directory and update metadata
        for doc in documents:
            new_file_path = doc.metadata['source']
            old_file_path = os.path.join(OLD_DOCUMENTS_DIR, os.path.basename(new_file_path))

            # Strip out 'transcription' and '.txt' to get the session ID
            base_name = os.path.basename(new_file_path)  # Get the file name without the directory path
            session_id = base_name.replace("transcription", "").replace(".txt", "")  # Remove the prefix and suffix
            doc.metadata['session_id'] = session_id

            session_txt = doc.page_content
            passed = send_text_to_api(session_txt, doc.metadata['session_id'])

            if not passed:
                return "There is an error with the header of the transcription file. Please make sure the header is correct."

            # Write the content to the old directory
            with open(old_file_path, 'w', encoding='utf-8') as f:
                f.write(doc.page_content)

            # Update the metadata to reflect the new path
            doc.metadata['source'] = old_file_path

            # Delete the file from the new directory
            os.remove(new_file_path)

        # Determine the length of the longest document
        max_document_length = max(len(doc.page_content) for doc in documents)

        max_document_length = min(max_document_length, 128000)

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = max_document_length + 1, chunk_overlap = max_document_length * 0.25)
        texts = text_splitter.split_documents(documents)

        # Ensure that each chunk has updated metadata
        for text in texts:
            text.metadata['source'] = os.path.join(OLD_DOCUMENTS_DIR, os.path.basename(text.metadata['source']))

        # Add new documents with updated metadata to the existing vector store
        vectordb.add_documents(texts)

        # Convert dictionary into string
        data = ""
        for key,value in topic_dict.items():
            data += f"{key},{value[0]},{value[1]}\n"

        header = "Topic,Frequency,AverageLength\n"
        to_write = header + data

        # Save the updated topic_dict to a CSV file
        with open('grouped_df_tableau.csv', 'w') as f:
            f.write(to_write)
    except:
        return "There is an error please make sure the files are txt format."


    return f"{len(documents)} documents added successfully."

"""Add header to transcription file"""
#This function might not be needed, because can just write description after transcribing
def add_Header_To_Transcription(file_path, session_id, description,user, date):
    """
    Function to add header to the transcription file

    Args:
    - file_path: The path to the transcription file
    - session_id: The session ID
    - description: The description of the session
    - user: The user who attended the session
    - date: The date of the session

    Returns:
    - None
    """
    # Read the existing transcription content
    with open(file_path, 'r') as file:
        transcription = file.read()

    # Prepare the metadata
    metadata = f"Therapy Session ID: {session_id}\nDescription: {description}\nClient: {user}\nDate: {date}\n\n"

    # Prepend the metadata to the transcription
    updated_transcription = metadata + transcription

    # Save the updated transcription back to the file
    with open(file_path, 'w') as file:
        file.write(updated_transcription)



"""Use Whisper to Transcribe Videos"""
import whisper
import subprocess

def get_video_length(video_path):
    """
    Function to get the length of the video or audio file

    Args:
    - video_path: The path to the video or audio file

    Returns:
    - length: The length of the video or audio file in minutes
    """
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    return float(result.stdout)

def is_valid_format(file_name):
    """
    Function to check whether the file format is valid

    Args:
    - file_name: The name of the file

    Returns:
    - valid: True if the file format is valid, False otherwise
    """
    valid_formats = ['.m4a', '.mp3', '.webm', '.mp4', '.mpga', '.wav', '.mpeg']
    return any(file_name.lower().endswith(ext) for ext in valid_formats)

def transcribe_video(video_name, session_id, description, user, date):
    """
    Function to transcribe video or audio

    Args:
    - video_name: The name of the video or audio file
    - session_id: The session ID
    - description: The description of the session
    - user: The user who attended the session
    - date: The date of the session

    Returns:
    - message: The message indicating the status of the transcription or the error message
    """

    if not is_valid_format(video_name):
        return "Invalid file format. Please keep the file in one of the following formats: .m4a, .mp3, .webm, .mp4, .mpga, .wav, .mpeg"

    try:
        model = whisper.load_model("base")
        video_path = os.path.join("Video", video_name)

        result = model.transcribe(video_path, fp16=False)
    except:
        return "There is no file with this name, please try again."

    # Define the directory where you want to save the transcription
    output_dir = "new_data"

    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the full path for the output file
    file_name = "transcription" + str(session_id) + ".txt"
    output_path = os.path.join(output_dir, file_name)

    #Compute video length
    video_duration = round(get_video_length(video_path)/60,2)

    # Prepare the metadata
    metadata = f"Therapy Session ID: {session_id}\nDescription: {description}\nClient: {user}\nDate: {date}\nLength of Session: {video_duration} minutes\n\n"

    #Full transcription
    full_transcription = metadata + result["text"]

    # Write the transcription to the file
    with open(output_path, "w") as f:
        f.write(full_transcription)

    add_document_to_db("Add new document")

    return f"Transcription have been added to the database."

"""Delete Chroma Vector DB"""
def delete_vector_db():
    """
    Function to delete the Chroma Vector DB

    Args:
    - None

    Returns:
    - None
    """
    # To cleanup, you can delete the collection
    vectordb = load_existing_vectorstore(persist_directory, embeddings)
    vectordb.delete_collection()


####################################################


from docx import Document
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma



from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI

from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv
import numpy as np
from pathlib import Path



QA_PROMPT = """Use the following pieces of content to answer the question at the end.
Answer strictly based on the content that is provided.
If you don't know the answer, just say you don't know, don't try to make up an answer.

<context>
{context}
</context>

Question: {input}
"""

def get_parent_dir():
    """
    Function to get the parent directory

    Args:
    - None

    Returns:
    - parent_path: The parent directory path
    """
    # Define the current path
    current_path = Path(__file__)
    # Get the parent directory
    parent_path = current_path.parent

    # Print the parent path
    return parent_path


def collect_data_as_one_docs(directory_path, output_file):
    """
    Function to collect all data in a directory into one document

    Args:
    - directory_path: The path to the directory containing the documents
    - output_file: The path to the output file

    Returns:
    - None
    """

    # Clear the content of the output file before writing
    with open(output_file, "w") as outfile:
        outfile.write("")

    with open(output_file, "a") as outfile:
        for filename in os.listdir(directory_path):
            if filename.endswith(".docx"):
                file_path = os.path.join(directory_path, filename)
                if os.path.isfile(file_path):
                    doc = Document(file_path)
                    full_text = []
                    for paragraph in doc.paragraphs:
                        full_text.append(paragraph.text)
                    content = "\n".join(full_text)
                    outfile.write(content + "\n\n")


def load_raw_data(filepath):
    """
    Function to load raw data from a file

    Args:
    - filepath: The path to the file containing the raw data

    Returns:
    - documents: The loaded documents

    """
    loader = TextLoader(file_path = filepath)
    documents = loader.load()

    return documents


def processing_docs(document):
    """
    Function to process documents into smaller chunks

    Args:
    - document: The document to process

    Returns:
    - texts: The processed texts
    """
    # split text into character
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2500)
    texts = text_splitter.split_documents(document)

    return texts


def text_2_embedding(texts, raw_data_filename):
    """
    Function to convert text to embeddings

    Args:
    - texts: The texts to convert to embeddings
    - raw_data_filename: The filename of the raw data

    Returns:
    - None
    """

    parent_path = get_parent_dir()
    vecdb_persist_folder = "VectorDB"
    vector_storing_dir = str(parent_path.joinpath(vecdb_persist_folder))

  
    raw_data_filename_without_ext = os.path.splitext(raw_data_filename)[0]
    persisit_dir = os.path.join(vector_storing_dir, raw_data_filename_without_ext)
    # embed and store texts
    
    embedding_model = OpenAIEmbeddings(api_key=os.environ['OPENAI_APIKEY'])
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embedding_model, persist_directory=persisit_dir
    )

    vectordb = None


####################

def retrievd_relevant_content(user_input_query,
                              vecdb_persist_folder,
                              knowledge_base_topic):
    """
    Function to retrieve relevant content

    Args:
    - user_input_query: The user input query
    - vecdb_persist_folder: The vector database persist folder
    - knowledge_base_topic: The knowledge base topic

    Returns:
    - retrievd_relevant_content: The retrieved relevant content
    """

    embedding_model = OpenAIEmbeddings(api_key=os.environ['OPENAI_APIKEY'])

    parent_path = get_parent_dir()
    # vecdb_persist_folder = "VectorDB"
    attach_vecdb_persist_dir = parent_path.joinpath(vecdb_persist_folder)

    # knowledge_base_topic = "theraphy_activity"
    vecdb_persist_dir =  str(attach_vecdb_persist_dir.joinpath(knowledge_base_topic))

    vectordb = Chroma(
        persist_directory = vecdb_persist_dir,
        embedding_function=embedding_model,
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 4}, score_threshold=0.95)
    retrievd_relevant_content = retriever.get_relevant_documents(user_input_query)
    

    return retrievd_relevant_content
   


def generate_answer(user_input_query,
                    relevant_content):
    """
    Function to generate an answer

    Args:
    - user_input_query: The user input query
    - relevant_content: The relevant content

    Returns:
    - chatbot_response: The chatbot response
    """
    
    prompt = ChatPromptTemplate.from_template(QA_PROMPT)
    text_generation_model = ChatOpenAI(temperature = 0.2, model = "gpt-4o-2024-05-13", 
                                       api_key=os.environ['OPENAI_APIKEY'])
    document_chain = create_stuff_documents_chain(text_generation_model, prompt)
    chatbot_response = document_chain.invoke({
        "input": user_input_query,
        "context": relevant_content
    })

    return chatbot_response


def send_code_to_api(user_input):
    """
    Function to send code to the API

    Args:
    - user_input: The user input

    Returns:
    - response: The response from the API
    """
    raw_data_persist_folder = "raw_data"
    doc_to_build_kb = "output.txt"
    vector_database_folder_name = "VectorDB"
    kb_topic = "theraphy_activity"

    parent_path = get_parent_dir()
    raw_data_persist_dir = str(parent_path.joinpath(raw_data_persist_folder))

    collect_data_as_one_docs(
        directory_path=raw_data_persist_dir, output_file=doc_to_build_kb
    )

    if os.path.isfile(doc_to_build_kb):
        with open(doc_to_build_kb, "r") as file:
            # print(f"Contents of {filename}:")
            loaded_data = load_raw_data(doc_to_build_kb)

            splitted_docs = processing_docs(loaded_data)
            text_2_embedding(texts=splitted_docs, raw_data_filename=kb_topic)

            refer_content = retrievd_relevant_content(user_input_query=user_input,
                                                    vecdb_persist_folder=vector_database_folder_name,
                                                    knowledge_base_topic=kb_topic
                                                    )
        
            response = generate_answer(user_input_query=user_input, relevant_content=refer_content)
            return response