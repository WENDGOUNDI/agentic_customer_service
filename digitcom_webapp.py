__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import time

from crewai import Agent, Task, Process, LLM, Crew
#from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource


#import warnings
#warnings.filterwarnings("ignore")

# Initialize the OpenAI API client
os.environ['OPENAI_API_KEY'] = st.secrets["bot_key"]

st.title("Aube Nouvelle AI Assistant")


# Set an LLM for the agent
llm = LLM(model="gpt-4o",
          temperature=0)

@st.cache_resource
def load_pdf_knowledge(filepath):
    print("Loading PDF knowledge...")
    pdf_source = PDFKnowledgeSource(file_paths=filepath)
    return pdf_source



pdf_source = load_pdf_knowledge(["Aube nouvelle.PDF",
                                 "Offres-de-formation-Universite-Aube-Nouvelle-Ouaga_250517_114215.pdf"])



# Create an agent with the knowledge store
university_agent = Agent(
    role="University Agent",
    goal="""ton role est de répondre aux questions sur l'universite Aube Nouvelle (U-Auben) en appelation francaise et New Dawn university en anglais qui autrefois
     s'appelait ISIG qui signifie Institut Superieur d'Informatique et de Gestion. Gnatan Isidore KINI en est le fondateur. Repond juste aux question posee, pas d'extra.
                        Si l'utilisateur utilise une autre langue que le francais, analyze la question ou la phrase pour la comprendre avant de repondre.""",
    backstory="""Tu es un expert en consultation, accompagnant toute personne qui souhaite en savoir plus sur l'université Aube Nouvelle (U-Auben) et ses offres de formation.
    Tu es capable de répondre à des questions sur l'université, ses programmes, ses valeurs et sa mission. Tu es également capable de fournir des informations sur les conditions d'admission et les procédures d'inscription.""",
    verbose=False,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description='''repond aux questions suivantes sur l'universite Aube Nouvelle (U-Auben) en appelation francaise et New Dawn university en anglais qui autrefois
     s'appelait ISIG qui signifie Institut Superieur d'Informatique et de Gestion: {question}''',
    expected_output="une reponse a la question.",
    agent=university_agent,
)

crew = Crew(
    agents=[university_agent],
    tasks=[task],
    verbose=False,
    process=Process.sequential,
    knowledge_sources=[pdf_source] )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Que puis je faire pour vous?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in crew.kickoff(inputs={"question": prompt}).raw:   
            full_response += response
            message_placeholder.markdown(full_response)
            time.sleep(0.05)
    print(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
