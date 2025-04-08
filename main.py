#Chat Log Parser and Structured Data Converter

import pandas as pd
import re

def parse_chat_log_with_human(file_path):
    """
    Reads a text file containing chat logs, parses it, and converts it into a structured DataFrame.

    Args:
    - file_path (str): Path to the text file containing the chat logs.

    Returns:
    - pd.DataFrame: A DataFrame containing structured chat log data.
    """
    # Open the file and read its lines
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Variables to store parsed data
    parsed_data = []
    lap_no, timestamp, customer, sentiment, category, intent, topic, pretrained, rag, system_response, human_response = (
        None, None, None, None, None, None, None, None, None, None, None)

    # Parse the data
    for line in data:
        # Detect new session
        if "--- New Session" in line:
            if lap_no is not None:
                parsed_data.append([lap_no, timestamp, customer, sentiment, category, intent, topic, pretrained, rag, system_response, human_response])
            lap_no, timestamp, customer, sentiment, category, intent, topic, pretrained, rag, system_response, human_response = (
                None, None, None, None, None, None, None, None, None, None, None)
            continue

        # Extract Lap No
        if "Lap No :" in line:
            if lap_no is not None:
                parsed_data.append([lap_no, timestamp, customer, sentiment, category, intent, topic, pretrained, rag, system_response, human_response])
            lap_no = re.search(r"Lap No : (\d+)", line)
            lap_no = int(lap_no.group(1)) if lap_no else None
            continue

        # Extract timestamp
        if re.match(r"\{'\d{4}-\d{2}-\d{2}", line):
            timestamp = line.strip()[1:-1]
            continue

        # Extract customer query
        if "Customer=>" in line:
            customer = line.split("Customer=>:")[1].strip()
            continue

        # Extract sentiment
        if "Sentiment=>" in line:
            sentiment = line.split("Sentiment=>:")[1].strip()
            continue

        # Extract category
        if "Category=>" in line:
            category = line.split("Category=>:")[1].strip()
            continue

        # Extract intent
        if "Intent=>" in line:
            intent = line.split("Intent=>:")[1].strip()
            continue

        # Extract Topic
        if "Topic=>" in line:
            topic = line.split("Topic=>:")[1].strip()
            continue

        # Extract PreTrained
        if "PreTrained=>" in line:
            pretrained = line.split("PreTrained=>:")[1].strip()
            continue

        # Extract RAG status
        if "RAG=>" in line:
            rag = line.split("RAG=>:")[1].strip()
            continue

        # Extract system response
        if "System=>" in line:
            system_response = line.split("System=>:")[1].strip()
            continue

        # Extract human response
        if "Human=>" in line:
            human_response = line.split("Human=>:")[1].strip()
            continue

    # Append the last entry
    if lap_no is not None:
        parsed_data.append([lap_no, timestamp, customer, sentiment, category, intent, topic, pretrained, rag, system_response, human_response])

    # Create DataFrame
    columns = ["Lap No", "Timestamp", "Customer Query", "Sentiment", "Category", "Intent", "Topic", "PreTrained", "RAG Status", "System Response", "Human Response"]
    df = pd.DataFrame(parsed_data, columns=columns)

    return df
#Parse and Display Chat Log Data in a DataFrame
file_path = '../input/chat_history (43).txt'

# Call the function
df = parse_chat_log_with_human(file_path)

# Display the DataFrame
print(df.head())

#Setup and Import Dependencies for LangChain Integration

!pip -q install openai
!pip -q install gradio
!pip -q install langchain-openai
!pip -q install langchain-core
!pip -q install langchain-community
!pip -q install sentence-transformers
!pip -q install langchain-huggingface
!pip -q install langchain-chroma
!pip -q install langchain_core.runnables
!pip -q install chromadb
!pip -q install pypdf
!pip install faiss-cpu
!pip install -U langchain
!pip install --upgrade langchain
#!pip install --upgrade langchain-retrievers

import os
import pickle
import openai
import numpy as np
import pandas as pd
import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import gradio as gr
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableMap, RunnableLambda

import os
from langchain_openai import ChatOpenAI

# Retrieve API key from Codespaces secret
openai_api_key = os.environ.get("OPENAI_API_KEY")  

# Initialize ChatOpenAI with the retrieved key
llm = ChatOpenAI(
    temperature=0.5,
    openai_api_key=openai_api_key,  # Use the retrieved key
    model_name="gpt-4o-mini"
)

prompt_template_llm_analysis = """
You are a performance analyst tasked with evaluating the responses of a language model (LLM) against human agent responses.
Below is a dataset containing chat interactions between customers, an LLM system, and a human agent. Analyze the data based on the specified criteria and provide a deterministic, structured report for performance evaluation. Follow the structure outlined below.

Dataset Description:
The dataset contains the following columns:

- New Session : <Agent Name> <timestamp>
- Lap No: The conversation sequence number.
- Timestamp: The time of each interaction.
- Customer Query: The question or message from the customer.
- Sentiment: The sentiment of the customer (Positive, Neutral, or Negative).
- Category: The general type of the query.
- Intent: The specific intent of the customer's query.
- RAG Status: Whether the system used Retrieval-Augmented Generation (RAG).
- System Response: The LLM's reply to the customer.
- Human Response: The human agent's reply to the customer.

Analysis Criteria:
Perform the following analysis on the dataset (please do not put any puntuation marks except comma (,), full stop (.) or semi column (:) in the output)

1. Accuracy and Relevance (Score: 30%):
    - Compare the LLM response with the human response for correctness, relevance, and clarity.
    - Evaluate whether the LLM captured the context and provided accurate information.
    - Scoring:
        - Full points (30%) for accurate and contextually relevant responses.
        - Partial points (15%) for partially correct or unclear responses.
        - No points (0%) for incorrect or irrelevant responses.

2. **Politeness and Tone (Score: 20%)**:
    - Compare the tone and politeness of the LLM's response with the human agent's.
    - Evaluate whether the LLM maintained professionalism and empathy in its responses.
    - **Scoring**:
        - Full points (20%) for consistently polite and empathetic responses.
        - Deduct points (-5% per instance) for responses lacking empathy or professionalism.

3. **Context Awareness (Score: 15%)**:
    - Analyze whether the LLM maintained continuity and understood the conversation context across multiple turns.
    - Compare the LLM's context awareness to the human agent's.
    - **Scoring**:
        - Full points (15%) for maintaining continuity and appropriately referencing prior interactions.
        - Partial points (7.5%) for minor lapses in context.
        - No points (0%) for completely losing context.

4. **Resolution Capability (Score: 20%)**:
    - Evaluate whether the LLM successfully resolved the customer's query.
    - Compare its resolution rate with that of the human agent.
    - **Scoring**:
        - Full points (20%) for resolving queries effectively.
        - Partial points (10%) for partial resolutions.
        - No points (0%) for unresolved queries.

5. **Sentiment Alignment (Score: 10%)**:
    - Assess whether the LLM aligned its tone and content with the sentiment expressed by the customer.
    - Compare the LLM's sentiment handling with the human agent's approach.
    - **Scoring**:
        - Full points (10%) for effectively addressing sentiment shifts.
        - Partial points (5%) for neutral handling of sentiment.
        - No points (0%) for worsening sentiment or ignoring customer emotions.

6. **Efficiency and Turn Optimization (Score: 5%)**:
    - Analyze the number of turns required to resolve queries for the LLM compared to the human agent.
    - Evaluate whether the LLM provided concise responses to minimize back-and-forth interactions.
    - **Scoring**:
        - Full points (5%) for concise and efficient interactions.
        - Deduct points (-1% per unnecessary turn).

###Output Format:
Provide the analysis in the following deterministic structure:

**LLM Performance Analysis Report: **

Metric,Score,Feedback
Query Resolution,<Score>,<Feedback>
Politeness,<Score>,<Feedback>
System vs. Human Accuracy,<Score>,<Feedback>
RAG Utilization,<Score>,<Feedback>
Sentiment Shift,<Score>,<Feedback>
Resolution Efficiency,<Score>,<Feedback>
Response Time,<Score>,<Feedback>

Overall Score and Feedback
- Score: [X%]
- Final Feedback: [Summary of LLM performance, areas of strength, and suggestions for improvement.]

### Here’s the data:
{dataframe}
"""
prompt_template_agent_analysis = """
You are a performance analyst tasked with evaluating chat support logs. Below is a dataset containing chat interactions between customers, a chatbot system, and a human agent. Analyze the data based on the specified criteria and provide a deterministic, structured report for performance evaluation. Follow the structure outlined below.

Dataset Description:
The dataset contains the following columns:

- New Session : <Agent Name> <timestamp>
- Lap No: The conversation sequence number.
- Timestamp: The time of each interaction.
- Customer Query: The question or message from the customer.
- Sentiment: The sentiment of the customer (Positive, Neutral, or Negative).
- Category: The general type of the query.
- Intent: The specific intent of the customer's query.
- RAG Status: Whether the system used Retrieval-Augmented Generation (RAG).
- System Response: The chatbot's reply to the customer.
- Human Response: The human agent's reply to the customer.

Analysis Criteria:
Perform the following analysis on the dataset (please do not put any puntuation marks except comma (,), full stop (.) or semi column (:) in the output)

1. **Query Resolution** (Score: 30%):
    - Was the customer query resolved based on the System Response and Human Response?
    - Highlight unresolved queries and provide feedback.
    - **Scoring**:
        - Full points (30%) for resolved queries.
        - Partial points (15%) for partially resolved queries.
        - No points (0%) for unresolved queries.

2. **Politeness** (Score: 20%):
    - Analyze the Human Response to determine whether the agent responded politely.
    - Provide feedback for laps where the response lacked politeness.
    - **Scoring**:
        - Full points (20%) for consistent politeness.
        - Deduct points (-5% per instance) for impolite responses.

3. **System vs. Human Accuracy** (Score: 20%):
    - Compare how often the system provided the correct response versus when the human agent had to correct it.
    - Identify patterns where the system frequently needed correction or where the human agent intervened.
    - **Scoring**:
        - Full points (20%) for accurate system responses and necessary corrections by the human agent.
        - Deduct points for system errors (-5% per error corrected by the agent).

4. **RAG Utilization** (Score: 10%):
    - Analyze how often RAG was invoked and whether it correlated with higher accuracy or better resolution.
    - Determine if invoking RAG reduced human intervention.
    - **Scoring**:
        - Full points (10%) for frequent and effective RAG usage.
        - Partial points (5%) for infrequent or ineffective usage.

5. **Sentiment Shift** (Score: 10%):
    - Track the sentiment of the customer throughout the conversation.
    - Report whether sentiment improved, stayed neutral, or became negative.
    - **Scoring**:
        - Full points (10%) for improving customer sentiment.
        - Partial points (5%) for neutral sentiment.
        - No points (0%) for worsening sentiment.

6. **Resolution Efficiency** (Score: 5%):
    - Measure how quickly queries were resolved (number of turns).
    - Report whether higher system accuracy or RAG utilization led to faster resolutions.
    - **Scoring**:
        - Full points (5%) for faster resolutions in fewer turns.
        - Deduct points (-1% per extra turn).

7. **Response Time** (Score: 5%):
    - Analyze the response time of the human agent.
    - Determine if longer response times correlated with unresolved queries or negative sentiment.
    - **Scoring**:
        - Full points (5%) for fast responses.
        - Deduct points for delays (-1% per delayed response).

### Output Format:
Provide the analysis in the following deterministic structure.
Support Performance Analysis: <Agent Name>  <timestamp>

Metric,Score,Feedback
Query Resolution,<Score>,<Feedback>
Politeness,<Score>,<Feedback>
System vs. Human Accuracy,<Score>,<Feedback>
RAG Utilization,<Score>,<Feedback>
Sentiment Shift,<Score>,<Feedback>
Resolution Efficiency,<Score>,<Feedback>
Response Time,<Score>,<Feedback>

Overall Score and Feedback
- Score: [X%]
- Final Feedback: [Summary of performance and recommendations for future training]


### Here’s the data:
{dataframe}
"""

import re

def get_prompt_template(file_path, prompt_1, prompt_2):
    with open(file_path, 'r') as file:
        for line in file:
            # Detect the first 'New Session' line
            if "--- New Session :" in line:
                # Extract the agent name
                agent_name_match = re.search(r"--- New Session :\s+(\w+)", line)
                if agent_name_match:
                    agent_name = agent_name_match.group(1)
                    # Return the appropriate prompt
                    return prompt_1 if agent_name == "Shiva" else prompt_2
    # Default return if no 'New Session' is found
    return prompt_2

# Define the file path
file_path = "../input/chat_history (43).txt"

# Define the prompts
prompt_1 = prompt_template_llm_analysis
prompt_2 = prompt_template_agent_analysis

# Get the appropriate prompt template
prompt_template = get_prompt_template(file_path, prompt_1, prompt_2)

# Print the result
print(f"Selected Prompt Template: {prompt_template}")

#The prompt object will be used to generate prompts with the specified template, dynamically filling in the dataframe variable with actual data.

prompt = PromptTemplate(
    input_variables=["dataframe"],
    template=prompt_template
)

from langchain.chains import LLMChain

# Set up the LLM chain with the prompt template and model
chain = LLMChain(llm=llm, prompt=prompt)

# Convert dataframe to string for input
df_str = df.to_string()

# Run the chain
performance_analysis = chain.run(dataframe=df_str)

# Print the performance analysis
print(performance_analysis)

# This will set up a Gradio interface to generate and download performance evaluation reports for human agents and llm agents.
# It includes a dropdown to select an agent, a button to generate the report, a textbox to display the analysis, and a link to download the report as a CSV.

import gradio as gr

def update_title(agent_name):
    if agent_name == "Shiva":
        return "<div style='text-align: center; font-size: 24px; font-weight: bold;'>Performance Evaluation of LLM Agent</div>"
    else:
        return "<div style='text-align: center; font-size: 24px; font-weight: bold;'>Performance Evaluation of Human Agent</div>"

with gr.Blocks() as demo:

        title_display = gr.HTML("")

        agent_name = gr.Dropdown(choices=["Mani", "Kamala", "Katrina", "Parul", "Shiva", "Rahul", "Sita", "Rati"], label="Select Agent Name")

        # Button to generate the performance evaluation report
        report_button = gr.Button("Generate Performance Evaluation Report")

        # Textbox to display the performance analysis
        analysis_textbox = gr.Textbox(lines=8, label="Performance Analysis", interactive=False)

        # Link to download the CSV report
        export_link = gr.File(label="Download CSV Report")

        # Function to generate and save the CSV report
        def generate_performance_report(agent_name):

            # Set up the LLM chain with the prompt template and model
            chain = LLMChain(llm=llm, prompt=prompt)

            # Convert dataframe to string for input
            df_str = df.to_string()

            # Run the chain
            performance_analysis = chain.run(dataframe=df_str)

            # Print the performance analysis
            print(performance_analysis)

            # Convert the analysis to a DataFrame
            lines = performance_analysis.strip().split('\n')
            data = [line.split(',', 2) for line in lines[1:]]
            df2 = pd.DataFrame(data)

            # Save the DataFrame to a CSV file
            temp_file_path = "/tmp/performance_report.csv"
            df2.to_csv(temp_file_path, index=False, header=False)

            print(f"Report saved at: {temp_file_path}")

            return performance_analysis, temp_file_path

        agent_name.change(
        fn=update_title,
        inputs=agent_name,
        outputs=title_display,  # Update the HTML component
        )

        # Link the button click to the generate_performance_report function
        report_button.click(
          fn=generate_performance_report,
          inputs=agent_name,
          outputs=[analysis_textbox, export_link]
        )

demo.launch(share=True, debug=True)
