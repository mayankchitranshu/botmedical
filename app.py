from flask import Flask,render_template,jsonify,request
frm src import healper.py


from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone
import pinecone
from langchain.document_loaders import PyMuPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.llms import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI



def load_pdf(data):
    loader=DirectoryLoader(data,glob="*.pdf",loader_cls=PyMuPDFLoader)

    documents=loader.load()

    return documents

extracted_data=load_pdf("data/")


def text_split(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_splitter.split_documents(extracted_data)

    return text_chunks

text_chunks=text_split(extracted_data)

import os
os.environ["OPENAI_API_KEY"]="sk-proj-EV1hw8qZlFkj8MUAogeXT3BlbkFJsAarOjFKBDuJ67kmx4Zy"
os.environ["PINECONE_API_KEY"]="81754309-acf1-4ac8-bba0-28f4da9d5426"

embedding=OpenAIEmbeddings()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY','81754309-acf1-4ac8-bba0-28f4da9d5426')

PINECONE_API_KEY="81754309-acf1-4ac8-bba0-28f4da9d5426"
index_name='medical-chatbot'
index=pinecone.Index(api_key='81754309-acf1-4ac8-bba0-28f4da9d5426',host='https://medical-chatbot-x603g1g.svc.aped-4627-b74a.pinecone.io')

docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embedding=embedding,index_name=index_name)

docsearch=Pinecone.from_existing_index(index_name,embedding)
query1="What are allergies?"
docs=docsearch.similarity_search(query1,k=3)
print("Result",docs)

prompt_template="""
Use the following pieces of information to answer the user's question.
If you don't know the answer,just say that you don't know ,don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

PROMPT=PromptTemplate(input_variables=["context","question"],template=prompt_template)
chain_type_kwargs={"prompt":PROMPT}

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.5,
    openai_api_key="sk-proj-EV1hw8qZlFkj8MUAogeXT3BlbkFJsAarOjFKBDuJ67kmx4Zy"
)

qa=RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=docsearch.as_retriever(),return_source_documents=True,chain_type_kwargs=chain_type_kwargs)

import os


app=Flask(__name__)




@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get",methods=["GET","POST"])
def chat():
    msg=request.form["msg"]
    input=msg
    print(input)
    result=qa({"query":input})
    print("Response :", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(debug=True)