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

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts.chat import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""使用下面的语料来回答本模板最末尾的问题。如果你不知道问题的答案，直接回答 "我不知道"。 
        以下是语料：
<context>
{context}
</context>

Question: {input}""")

#创建检索链
document_chain = create_stuff_documents_chain(chat, prompt)

retriever = vectorstore.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import ConversationChain

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa = ConversationalRetrievalChain.from_llm(chat, vectorstore.as_retriever(), memory=memory)

questions = [
  "小王在签合同时没认真看格式条款，对方也未做出说明，事后小王觉得自己遭遇“霸王条款”，相关条款有效吗？",
  "他后续应该怎么办？"
]
for question in questions:
        print(question)
        answer = qa.invoke(question)["answer"]
        print(answer)
        print()
