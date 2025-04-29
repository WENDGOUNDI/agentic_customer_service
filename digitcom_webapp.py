import streamlit as st
import os
import time

from crewai import Agent, Task, Process, LLM, Crew
from crewai.knowledge.source.csv_knowledge_source import CSVKnowledgeSource
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource


#import warnings
#warnings.filterwarnings("ignore")

# Initialize the OpenAI API client
os.environ['OPENAI_API_KEY'] = st.secrets["bot_key"]


st.title("DigitCom AI Assistant")


# Set an LLM for the agent
llm = LLM(model="gpt-4o-mini",
          temperature=0)

@st.cache_resource
def load_pdf_knowledge(filepath):
    print("Loading PDF knowledge...")
    pdf_source = PDFKnowledgeSource(file_paths=[filepath])
    return pdf_source

@st.cache_data
def load_csv_data(filepath):
    print("Loading CSV data...")
    # Assuming your CSVKnowledgeSource returns some processed data
    csv_source = CSVKnowledgeSource(file_paths=[filepath])
    return csv_source # Or similar method

pdf_source = load_pdf_knowledge("company_description.pdf")
csv_source = load_csv_data("ventes_digitcom_2 - Sheet1.csv")


# Create an agent with the knowledge store
digicom_agent = Agent(
    role="Company Agent",
    goal="ton role est de répondre à des questions sur DigitCom, une entreprise de vente de produits digitaux/ecommerce. reepond juste qux question pose, pas d'extra",
    backstory="""Tu es un expert a repondre aux questions liees a DigitCom.""",
    verbose=False,
    allow_delegation=False,
    llm=llm,
)
task = Task(
    description="repond aux questions suivantes sur DigitCom: {question}",
    expected_output="une reponse a la question.",
    agent=digicom_agent,
)

crew = Crew(
    agents=[digicom_agent],
    tasks=[task],
    verbose=False,
    process=Process.sequential,
    knowledge_sources=[pdf_source, csv_source], )


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What can I do for you?"):
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