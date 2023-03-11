#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st 


# In[4]:


import PyPDF2


# In[7]:


import pdfplumber


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


from sklearn.metrics.pairwise import cosine_similarity


# In[10]:


st.title("Candidate Selection Tool")

st.subheader("NLP Based Resume Screening")

st.caption("Aim of this project is to check whether a candidate is qualified for a role based his or her education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.")

uploadedJD = st.file_uploader("Upload Job Description", type="pdf")

uploadedResume = st.file_uploader("Upload resume",type="pdf")

click = st.button("Process")


# In[11]:


try:
    global job_description
    with pdfplumber.open(uploadedJD) as pdf:
        pages = pdf.pages[0]
        job_description = pages.extract_text()

except:
    st.write("")
    
    
try:
    global resume
    with  pdfplumber.open(uploadedResume) as pdf:
        pages = pdf.pages[0]
        resume = pages.extract_text()
except:
    st.write("")


# In[12]:


#logic
def getResult(JD_txt,resume_txt):
    content = [JD_txt,resume_txt]

    cv = CountVectorizer()

    matrix = cv.fit_transform(content)

    similarity_matrix =  cosine_similarity(matrix)

    match = similarity_matrix[0][1] * 100

    return match


# In[13]:


#button 

if click:
    match = getResult(job_description,resume)
    match = round(match,2)
    st.write("Match Percentage: ",match,"%")

st.caption(" ~ made by Tejas")


# In[ ]:




