import streamlit as st
import openai
import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

st.title("Measuring JDs against the JTM")

#########################################

openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_prompt(jtm_job, jtm):
    prompt = f"""

            Imagine you are a hiring manager in 2030 and hiring \
            {jtm_job}. Write a job description based only on information \
            from this extract of a report detailing 
            how the {jtm_job} will change in the future.The \
            job description should include a description of the role, \
            responsibilities and skills required.

            Report: {jtm}

            """
    return prompt

embedding_model = SentenceTransformer('paraphrase-albert-small-v2')

if 'jtm_jd' not in st.session_state:
    st.session_state['jtm_jd'] = None

#########################################

tab1, tab2, tab3 = st.tabs(["Intro", "Part 1: JTM", "Part 2: JDs"])

with tab1:
    st.write("insert problem statement and application of this approach")

with tab2:
    jtm_job = st.text_input(label="Name of Job")
    jtm = st.text_area(label="Enter the JTM extract")

    if st.button("Generate JD"):
        prompt = get_prompt(jtm_job, jtm)
        st.session_state.jtm_jd = get_completion(prompt)
        st.info(st.session_state.jtm_jd)

with tab3:

    col1, col2 = st.columns([0.6, 0.4])

    jd1 = col1.text_area("Enter the Job Description 1")
    jd2 = col1.text_area("Enter the Job Description 2")
    jd3 = col1.text_area("Enter the Job Description 3")

    col1.write(st.session_state.jtm_jd)

    if col1.button("Compare Against JTM"):

        jd1_embed = embedding_model.encode(jd1)
        jd2_embed = embedding_model.encode(jd2)
        jd3_embed = embedding_model.encode(jd3)
        jtm_embed = embedding_model.encode(st.session_state.jtm_jd)
        
        jd1_score = np.dot(jtm_embed, jd1_embed)
        jd2_score = np.dot(jtm_embed, jd2_embed)
        jd3_score = np.dot(jtm_embed, jd3_embed)

        df = pd.DataFrame([["JD 1", "JD 2", "JD 3"],
                           [jd1_score, jd2_score, jd3_score]]).transpose()
        
        df.columns = ["JD", "Similarity Score"]
        
        col2.dataframe(df)       
 
