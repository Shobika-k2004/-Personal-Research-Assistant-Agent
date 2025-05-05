import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import init_chat_model
from tools import search_papers
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Set up the page
st.title("Research Assistant")

# Initialize the model
llm = init_chat_model(
    model="llama3-8b-8192",
    model_provider="groq",
    temperature=0.7,
    api_key="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"
)

# Create a summarization prompt
summary_prompt = PromptTemplate(
    input_variables=["papers"],
    template="""
    Please analyze and summarize the following academic papers. Focus on:
    1. Main findings and conclusions
    2. Key methodologies used
    3. Important statistics or data points
    4. How these papers relate to each other
    5. Gaps or areas for future research

    Papers to summarize:
    {papers}

    Provide a coherent overview that ties these papers together and highlights their significance.
    """
)

# Initialize the summarization chain
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Search interface
st.markdown("###  Enter Your Research Query")
query = st.text_area("What would you like to research?", 
                    placeholder="e.g., 'Recent developments in quantum computing' or 'Impact of climate change on biodiversity'")

# Additional filters
col1, col2 = st.columns(2)
with col1:
    year_range = st.slider("Publication Year Range", 2000, 2024, (2020, 2024))
with col2:
    num_papers = st.slider("Number of Papers to Analyze", 3, 10, 5)

if st.button("Research"):
    if query:
        with st.spinner(" Searching and analyzing papers..."):
            try:
                # Search for papers
                url = "https://api.semanticscholar.org/graph/v1/paper/search"
                params = {
                    "query": query,
                    "limit": num_papers,
                    "fields": "title,abstract,url,year,authors,citationCount",
                    "year": f"{year_range[0]}-{year_range[1]}"
                }
                response = requests.get(url, params=params)
                papers = response.json().get("data", [])

                if papers:
                    # Display individual papers
                    st.markdown("### Found Papers")
                    papers_text = ""
                    for i, paper in enumerate(papers, 1):
                        with st.expander(f"{i}. {paper['title']} ({paper.get('year', 'N/A')})"):
                            st.write(f"**Authors:** {', '.join([author['name'] for author in paper.get('authors', [])])}")
                            st.write(f"**Abstract:** {paper.get('abstract', 'No abstract available.')}")
                            st.write(f"**Citations:** {paper.get('citationCount', 0)}")
                            st.write(f"**URL:** {paper['url']}")
                            papers_text += f"\n\nPaper {i}:\nTitle: {paper['title']}\nAbstract: {paper.get('abstract', '')}\n"

                    # Generate and display summary
                    st.markdown("###  Research Summary")
                    summary = summary_chain.run(papers=papers_text)
                    st.write(summary)

                    # Add download button for the summary
                    st.download_button(
                        label="Download Summary",
                        data=summary,
                        file_name="research_summary.txt",
                        mime="text/plain"
                    )
                else:
                    st.warning("No relevant papers found. Try adjusting your search criteria.")
                    
            except Exception as e:
                st.error(f"Error during research: {str(e)}")
    else:
        st.warning("Please enter a research query")

# Add footer
st.markdown("---")
