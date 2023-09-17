"""
TODOS:
1. multiple rounds of chat
2. more prompts

https://www.youtube.com/watch?v=3yPBVii7Ct0&ab_channel=SamWitteveen
https://learn.deeplearning.ai/langchain-chat-with-your-data/lesson/7/chat

This is a demo for QA system using langchain.
Users can upload a pdf file and index it.
Users can ask questions and get answers from the indexed pdf file.

Dependencies:
The demo is based on streamlit, so you need to install streamlit first.
The demo is based on openai, so you need to install openai first.
The demo is based on chromadb, so you need to install chromadb first. The chromadb is used for indexing the document.
The demo is based on langchain, so you need to install langchain first. The langchain is used for QA system.
The demo is based on pypdf2, so you need to install pypdf2 first. The pypdf2 is used for parsing the pdf file.

QA:
There are three kind of search methods:
1. simple search without retriever
    # simple search without retriever
    # docs = vectordb_load.similarity_search(question,k=5)

2. make a simple retriever
    # make a simple retriever
    # retriever = vectordb_load.as_retriever(search_kwargs={"k": 5})
    # st.write(retriever.search_type)
    # docs = retriever.get_relevant_documents(question)
    # # print the results
    # for doc in docs:
    #     st.write(doc.page_content)
    #     st.write(doc.metadata)

3. make a compressor retriever
The method 3 is used in this demo.
"""

import streamlit as st
import os
import openai


from tools import (
    pretty_print_docs
)

# load pdf
from langchain.document_loaders import PyPDFLoader
# split text into paragraphs
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
# embedding
from langchain.embeddings.openai import OpenAIEmbeddings

# pip install chromadb
from langchain.vectorstores import Chroma

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

from langchain.chains import RetrievalQA

from langchain.prompts import PromptTemplate

from langchain.retrievers.merger_retriever import MergerRetriever

from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



# after uploading and indexing the file, the file and its index will be saved in the data folder
data_path = './qa_data'
if not os.path.exists(data_path):
    os.makedirs(data_path)

# clear submit, but why?
def clear_submit():
    st.session_state["submit"] = False

# sidebar for openai api key
st.sidebar.title("QA GPT")
st.sidebar.markdown("please enter your OpenAI API key")
api_key_input = st.sidebar.text_input(
    "OpenAI API Key",
    placeholder="Paste your OpenAI API key here (sk-...)",
    help="You can get your API key from https://platform.openai.com/account/api-keys.",
    value=st.session_state.get(
        "OPENAI_API_KEY", "sk-"),
)

# save api key
if api_key_input:
    st.session_state["OPENAI_API_KEY"] = api_key_input
    openai.api_key = api_key_input
    # embedding
    embedding = OpenAIEmbeddings(openai_api_key=api_key_input)

st.sidebar.markdown("---")

# sidebar for document upload
st.sidebar.title("📁 index new file")
# upload file for index
uploaded_file = st.sidebar.file_uploader(
    # pypdf cannot process scanned PDF, so OCR should be done before.
    "Upload a pdf(after OCR if it is scanned) file",
    type=["pdf"],
    help="Scanned documents are not supported yet!",
    on_change=clear_submit,
)

# set category
category = st.sidebar.text_input(
    "category",
    placeholder="Enter the category of the document",
    help="Category cannot be none!",
    )

# upload file
if uploaded_file is not None and category is not None and category != "":
    # generate path
    filename = uploaded_file.name
    category_path = os.path.join(data_path, category)
    file_save_path = os.path.join(category_path, filename)
    index_db_path = os.path.join(
        category_path, filename.replace(".", "_") + "_index/")
    doc_path = os.path.join(
        category_path, filename.replace(".", "_") + ".jsonl")
    folder_path = os.path.join(
        category_path, filename.replace(".", "_"), "")
    
    if os.path.exists(doc_path) and os.path.exists(index_db_path):
        st.sidebar.error("File index exists, please change the name!")
        raise ValueError("File index exists, please change the name!")
    
# overwrite index
overwrite_index = st.sidebar.checkbox("Overwrite index", value=False)

# index file
if st.sidebar.button("index file"):
    # extract
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".pdf"):
            
            # create category folder if not exists
            if not os.path.exists(category_path):
                os.makedirs(category_path)
            # save the uploaded file to category folder
            with open(file_save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        else:
            raise ValueError("File type not supported!")
        
    # parse pdf by langchain document loader
    loader = PyPDFLoader(file_save_path)
    pages = loader.load()
    # st.write(len(pages))
    # st.write(pages[0])
    # st.write(pages[0].page_content)   

    # split text into paragraphs
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)
    splits = text_splitter.split_documents(pages)

    st.write(len(splits))
    st.write(splits[0].page_content)

    # vector store
    # if index_db_path is not empty, prompt "index exists, do you want to overwrite?"
    if index_db_path is not None:
        if os.path.exists(index_db_path):
            if overwrite_index:
                st.sidebar.info("Indexing document... This may take a while⏳")
                vectordb = Chroma.from_documents(
                documents=splits,
                embedding=embedding,
                persist_directory=index_db_path
            )
                st.sidebar.success("Indexing done!✅")
                # save index for later use
                vectordb.persist()
                st.sidebar.success("Index saved!✅")
            else:
                st.sidebar.error("Index exists, please change the name!")
                raise ValueError("Index exists, please change the name!")
        else:
            
            st.sidebar.info("Indexing document... This may take a while⏳")
            vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=index_db_path
        )
            st.sidebar.success("Indexing done!✅")
            # create index folder if not exists
            if not os.path.exists(index_db_path):
                os.makedirs(index_db_path)
            # save index for later use
            vectordb.persist()
            st.sidebar.success("Index saved!✅")

# select category
# get category list, which is the folder list under data_path
if os.path.exists(data_path):
    folder_list = os.listdir(data_path)
    category_list =[i for i in folder_list if os.path.isdir(os.path.join(data_path, i))]

select_category = st.selectbox("Select category", category_list)

if select_category is not None:
    select_category_path = os.path.join(data_path, select_category)
else:
    select_category_path = None

# get db list
select_dbs = None
if select_category_path is not None:
    if os.path.exists(select_category_path):
        file_list = os.listdir(select_category_path)
        db_list =[i for i in file_list if i.endswith(".pdf")]

    # multiple select from db_list
    select_dbs = st.multiselect("Select files", db_list)

# get the selected db index_db_path
if select_dbs is not None:
    if len(select_dbs) > 0:
        # get multiple index_db_path
        select_db_paths = [os.path.join(select_category_path, i.replace(".", "_") + "_index/") for i in select_dbs]
    else:
        select_db_paths = None


# load index from disk
if st.button("test load index"):
    for path in select_db_paths:
        vectordb_load = Chroma(persist_directory=path, embedding_function=embedding)
        if vectordb_load is not None:
            st.write(path, "exists")
            st.write("The content length is", len(vectordb_load.get()["ids"]))


# ask question
st.title("🔎 Ask a question")
question = st.text_input(
    "Question",
    placeholder="Enter your question here",
    help="Enter your question here",
    )

# model type from a streamlit selectbox
models = {"gpt-3.5-turbo-16k": 16384, "gpt-3.5-turbo": 4096, "gpt-4-32k": 32768, "gpt-4": 8192}
llm_name = st.selectbox("Select model type", list(models.keys()))

# temparature
temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)

# select retrieval mode
retrieval_mode = st.selectbox("Select retrieval mode", ["simple retriever", "compressor retriever"])

# load model
llm_qa = ChatOpenAI(openai_api_key=api_key_input, model_name=llm_name, temperature=temperature)
llm = OpenAI(openai_api_key=api_key_input, temperature=0)



# Build prompt

templates = {}
template1 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. 
{context}
Question: {question}
Helpful Answer:"""
templates["3-sentence answer, stick to resources"] = template1
template2 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use one paragraph.
{context}
Question: {question}
Helpful Answer:"""
templates["1-paragraph answer, stick to resources"] = template2
template3 = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. You can be as detailed as necessary.
{context}
Question: {question}
Helpful Answer:"""
templates["detailed answer, stick to resources"] = template3

# select template
select_prompt = st.selectbox("Select template", list(templates.keys()))

QA_CHAIN_PROMPT = PromptTemplate.from_template(templates[select_prompt])

# search mode 1: get one single answer from multiple files
if st.button("Search mode 1: get one single answer from multiple files"):
    # make a compressor retriever
    # Wrap our vectorstore
    # llm = OpenAI(temperature=0, openai_api_key=api_key_input)
    # st.write(api_key_input)
    compressor = LLMChainExtractor.from_llm(llm)
    # st.write(compressor)

    # load index from disk
    # if only one index_db_path is selected, load it
    if len(select_db_paths) == 1:
        vectordb_load = Chroma(persist_directory=select_db_paths[0], embedding_function=embedding)
        base_retriever = vectordb_load.as_retriever(search_kwargs={"k": 5})
    elif len(select_db_paths) > 1:
        # initiate multiple retrievers
        retrievers = []
        # make a simple retriever
        for path in select_db_paths:
            vectordb_load = Chroma(persist_directory=path, embedding_function=embedding)
            retriever = vectordb_load.as_retriever(search_kwargs={"k": 5})
            retrievers.append(retriever)
        # merge multiple index_db_path
        # merge retrievers to support multiple retrievers, index in multiple chroma dbs
        base_retriever = MergerRetriever(retrievers=retrievers)

    compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

    # st.write(compression_retriever)
    # # Get the relevant documents
    # compressed_docs = compression_retriever.get_relevant_documents(question)
    # st.write(compressed_docs)
    # # print the results
    # pretty_print_docs(compressed_docs)

    # set the retriever
    if retrieval_mode == "simple retriever":
        retriever = base_retriever
    elif retrieval_mode == "compressor retriever":
        retriever = compression_retriever

    # Run chain
    qa_chain = RetrievalQA.from_chain_type(
        llm_qa,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    result = qa_chain({"query": question})

    st.write(result["result"])
    pretty_print_docs(result["source_documents"])


# search mode 2: get multiple answers from each file and then synthesize them
if st.button("Search mode 2: get multiple answers from each file and then synthesize them"):
    # make a compressor retriever
    # Wrap our vectorstore
    # llm = OpenAI(temperature=0, openai_api_key=api_key_input)
    # st.write(api_key_input)
    compressor = LLMChainExtractor.from_llm(llm)
    # st.write(compressor)

    # if only one index_db_path is selected, load it
    results = []
    for path in select_db_paths:
        # load index from disk
        vectordb_load = Chroma(persist_directory=path, embedding_function=embedding)
        # make a simple retriever
        base_retriever = vectordb_load.as_retriever(search_kwargs={"k": 5})       
        # make a compressor retriever
        compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
        
        # set the retriever
        if retrieval_mode == "simple retriever":
            retriever = base_retriever
        elif retrieval_mode == "compressor retriever":
            retriever = compression_retriever

        # Run chain
        qa_chain = RetrievalQA.from_chain_type(
            llm_qa,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )

        result = qa_chain({"query": question})
        results.append(result)
    
    # use the results as new context to ask the same question
    # create a new retriever from the results
    chat = ChatOpenAI(temperature=temperature, openai_api_key=api_key_input, model_name=llm_name)
    template = ""
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = templates[select_prompt]
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    synthetic_answer = chat(
        chat_prompt.format_prompt(
            context="\n".join([r["result"] for r in results]), question=question
        ).to_messages()
    )

    st.write(synthetic_answer.content)
    for result in results:
        pretty_print_docs(result["source_documents"])


        

        