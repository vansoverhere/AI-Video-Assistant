from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import  RunnableLambda

import os

def get_llm():
    return ChatGoogleGenerativeAI(model = "gemini-3.1-flash-lite", google_api_key=os.getenv("GEMINI_API_KEY"), temperature=0.3 )


def split_transcript(transcript:str)->list :
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 3000,
        chunk_overlap = 200
    )
    return splitter.split_text(transcript)


def summarize(transcript : str) -> str : 
    llm = get_llm()

    map_prompt= ChatPromptTemplate.from_messages([
        ("system" , "Summarize this portion of a video transcript"),
        ("human", "{text}"),  
    ])

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk }) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages([
        ("system" , "You are an expert video summarizer.Combine these partial summaries"),
        ("human" , "{text}")
    ])

    combined_chain = (
        RunnableLambda(lambda x:{"text":x}) | combined_prompt | llm | StrOutputParser()
    )

    return combined_chain.invoke(combined)

def generate_title(transcipt : str) -> str:
    llm = get_llm()

    
    title_chain = (
        RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the video transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm
        |StrOutputParser()
    )

    return title_chain.invoke(transcipt[:2000])