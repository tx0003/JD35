import os
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
 
chat=ChatOpenAI(
     openai_api_key=os.environ["OPENAI_API_KEY"],
     model='gpt-3.5-turbo'
 )

#载入民法典pdf
loader=PyPDFLoader("民法典.pdf")
pages=loader.load_and_split()

#分割文本
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)
docs=text_splitter.split_documents(pages)

#利用embedding 模型对每一个文本片段进行向量化，并储存到向量数据库中
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

embed_model=OpenAIEmbeddings()
vectorstore=Chroma.from_documents(documents=docs,embedding=embed_model,collection_name="openai_embed") 

#增强
def augment_prompt(query:str):
    results=vectorstore.similarity_search(query,k=10)
    source_knowledge="\n".join([x.page_content for x in results])
    augmented_prompt= f"""Using the contexts below, answer the query.
    
    contexts:
    {source_knowledge}
    
    query:{query}"""
    return augmented_prompt
    
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
messages=[
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="knock knock."),
    AIMessage(content="Who is there"),
    HumanMessage(content="Orange"),
]

query=input("请提出你的问题：")

prompt=HumanMessage(
    content=augment_prompt(query)
)
messages.append(prompt)
res=chat(messages)
print(res.content)
