import streamlit as st
import openai
import numpy as np
import pandas as pd
import requests

from sentence_transformers import SentenceTransformer

st.title("Demo: Comparing JTM with JDs")

#########################################

openai.api_key = st.secrets["OPENAI_API_KEY"]
hf_token = st.secrets["HF_API_KEY"]

model_id = "sentence-transformers/all-mpnet-base-v2"
hf_api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
hf_header = {"Authorization": f"Bearer {hf_token}"}

embedding_model = SentenceTransformer('paraphrase-albert-small-v2')


#########################################

def get_completion(prompt, model="gpt-3.5-turbo-16k"):
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
            {jtm_job} in Singapore. Write a job description based only on information \
            from this extract of a report detailing 
            how the {jtm_job} will change in the future.The \
            job description should include a description of the role, \
            responsibilities and skills required.

            Do not include any footnote/endsnote at the end nor reference to the year 2030.

            Draft the JD as though as it will be published on a job portal.

            Report: {jtm}

            """
    return prompt

def get_HF_embedding(texts):
    response = requests.post(hf_api_url, 
                             headers=hf_header, 
                             json={"inputs": texts, "options":{"wait_for_model":True}})
    return response.json()

if 'jtm_jd' not in st.session_state:
    st.session_state['jtm_jd'] = "Please click `Generate JD`"

#########################################

tab1, tab2, tab3 = st.tabs(["Intro", "Part 1: JTM", "Part 2: JDs"])

with tab1:
    st.write("""

### Assumed Problem Statement 🔍

How might we measure the quality of jobs and programmes relative to the Job Transform Maps (JTMs) in order to identify and promote quality jobs for Singaporeans?

JTMs: https://www.wsg.gov.sg/home/employers-industry-partners/jobs-transformation-maps

### General Approach 🧭

1. Extract portions of JTMs that describe the future job
2. Generate a hypothetical JD based on the JTM extract using a large language model 
3. Calculate a similarity score between JTM-based JD (i.e. goal) against existing JDs (e.g. on MCF)

In this POC, we demonstrate the above using a simple front-end that ranks 3 user-provided JDs against the user-provided JTM extract.

This approach can generally scale to larger applications.

### Potential Applications 💻
                     
1. **Citizens:** Identify "JTM-aligned" jobs on MCF
2. **Employers:** Nudge job redesign efforts
3. **WSG and Sectoral Agencies:** Measure progress of JTM implementation over time
             
""")

with tab2:
    st.write("### Enter JTM")

    jtm_job = st.text_input(label="Name of Job",
                            value="Bank Relationship Manager"
                            )
    jtm = st.text_area(label="Enter the JTM extract",
                       value="""In Corporate Banking, the Relationship Managers (RMs)
role will be augmented by data analytics and automation
to focus on managing mid-sized corporate clients. Unlike
small enterprises, subsidiaries or FIs (where information is
available and accessible), the industry shared that managing
medium enterprises tends to be slightly more complex.
A new approach to segmentation (emphasising on behaviours
and needs, rather than revenue) will allow banks to gain
better understanding of the complexity of client tranches,
where ultimately the RM could prioritise their efforts on
the most valuable customers.
The future of this role will shift from reactive to more
proactive management of client accounts by leveraging
software/analytics that gather intelligence to identify cross
selling opportunities (this is not limited to FX, trade or cash
management). As with the role of Product Sales, digital
and self-service platforms have been implemented to support
the acquisition for simple accounts as well as other post-sales
activities. With the right tools in place, RMs can anticipate
different client needs and provide bespoke recommendations
that lead to a more meaningful exchange with their clients.

Skills Implication:
Whilst lateral thinking and advisory skills are inherent
for this role, future RMs will need to be proficient with
different tools, and understand how to leverage them to
further improve their engagement with clients. Rather
than merely understanding the theories, they also need
to be proficient in the application of data, specifically
learning how to interpret information and translate
insights into sound decisions or solutions.
"""
                       )

    if st.button("Generate JD"):
        prompt = get_prompt(jtm_job, jtm)
        st.session_state.jtm_jd = get_completion(prompt)

    st.write("### Generated JD")
    st.info(st.session_state.jtm_jd)

with tab3:

    col1, col2 = st.columns([0.6, 0.4])



    jd1 = col1.text_area("Job Description 1",
                         value="""Acquire and leading project financing cases. Initiate syndication loan, term loan and trade finance business and market the bank’s products to corporations in Singapore and the region.
    Have strong financial analysis skills, evaluate and process credit applications for timely approval, prepare loan/security documents, and promote customer active utilization of the facilities.
    Have better understanding of different corporate industries, initiate and participate in industry surveys and deliver industry investigation reports for Head Office.
    Have the ability to summarize corporate signature cases and prepare PPT and articles to promote among Head Office and domestic branches, have strong presentation skills.
    Explore and expand new corporate customer relationships
    Manage existing portfolio.
    Maintain customer relationships and organize customer events with the team.
    Communicate with Head Office and local active banks and other different stakeholders within banking industry, proactively to make sure deal run smoothly.
    Good understanding and knowledge of relevant compliance and risk regulations

Requirements:

    Recognised University Degree with min. 5-7 years’ experience in corporate banking, covering major corporate client relationships.
    Have strong financial analysis skills, project financing experience preferred.
    Positive working attitude, respond to the task assigned timely and responsibly and a good team player
""")

    
    jd2 = col1.text_area("Job Description 2",
                         value="""about the company
As a leading bank with both a huge local and global presence, our client is constantly expanding their team of Relationship Manager to provide a seamless banking experience.

about the job

    Take charge of managing and building a portfolio of medium to large-sized SMEs in the various industry.
    Develop well-structured trade solutions that meets customer’s needs
    Prepare credit proposals and oversee the implementation and disbursement of credit.
    Ensure timely completion of credit reviews.
    Provide top-notch customer service to clients.

about the manager/team
You will be reporting to the Team Head to further grow and increase your exposure.

skills and experience required
A degree is minimally required. You should come with 5+ years of experience in the relevant area and well-versed with Trade, Lending, CASA, FX and Banassurance products. Importantly, you should have experience in credit underwriting and financial analysis
""")

    jd3 = col1.text_area("Job Description 3",
                         value="""
Our client is a prominent bank that offers diverse financial services, including corporate and personal banking, investment banking, asset management, and insurance. With a wide network of branches and a significant presence both locally and internationally, the bank makes a substantial contribution to the financial sector. It is recognized for its financial stability, robust operations, and technological advancements, establishing itself as a major player in the global banking industry.

about the job

    Develop and maintain strong relationships with corporate clients involved in commodity trading and structured trade finance, understanding their needs and objectives
    Identify and pursue new opportunities in the commodity and structured trade finance sector, expanding the bank's portfolio and market share
    Design tailored trade finance solutions for clients, considering commodity trading intricacies, risk mitigation, and profitability enhancement
    Evaluate client creditworthiness, assess risks associated with commodity trading and structured finance, and make informed decisions on credit limits and terms
    Negotiate trade finance deal terms, ensuring compliance with regulations and addressing legal and risk-related aspects
    Work with internal stakeholders to execute trade finance transactions seamlessly, ensuring regulatory compliance and adherence to internal policies

skills and experience required

    Bachelor's degree or higher, preferably in Business or a related discipline
    At least 3 - 5 years of relevant experience in corporate banking
    Strong knowledge of corporate banking products and services including lending, cash management, trade finance, and treasury solutions
    Strong business acumen and market awareness, with the ability to identify and capitalize on business opportunities

""")


    jd1_score = 0
    jd2_score = 0
    jd3_score = 0

    if col1.button("Compare Against JTM"):

        # jd1_embed = get_HF_embedding(jd1)
        # jd2_embed = get_HF_embedding(jd2)
        # jd3_embed = get_HF_embedding(jd3)
        # jtm_embed = get_HF_embedding(jd3)

        jd1_embed = embedding_model.encode(jd1)
        jd2_embed = embedding_model.encode(jd2)
        jd3_embed = embedding_model.encode(jd3)
        jtm_embed = embedding_model.encode(jd3)
        

        jd_max = np.dot(jtm_embed, jtm_embed).round(1)
        jd1_score = np.round(np.dot(jtm_embed, jd1_embed) / jd_max, 2)
        jd2_score = np.round(np.dot(jtm_embed, jd2_embed) / jd_max, 2)
        jd3_score = np.round(np.dot(jtm_embed, jd3_embed) / jd_max, 2)

    df = pd.DataFrame([["JD 1", "JD 2", "JD 3"],
                       [jd1_score, jd2_score, jd3_score]]).transpose()
        
    df.columns = ["JD", "Similarity"]
        
    col2.dataframe(df)       
 
