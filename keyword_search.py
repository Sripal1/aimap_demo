import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from hybridSearch import HybridSearch
import plotly.express as px

st.set_page_config(
    page_title="AiMap researcher discovery",
    page_icon="",
    layout="wide"
)

st.markdown("""
    <style>
        .stAlert {
            padding: 0.5rem;
        }
        .stProgress .st-bo {
            height: 5px;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.2rem;
        }
        .small-text {
            font-size: 0.8rem;
            color: #666;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def loadEmbeddingModel():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def loadLLM():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    return tokenizer, model

@st.cache_resource
def initialize_search():
    df = pd.read_csv('profData.csv')
    df = df.dropna(subset=['Title', 'Abstract', 'AuthorKeywords'])
    
    df['concatenated'] = df['Title'] + ' ' + df['Abstract'] + ' ' + df['AuthorKeywords']
    concatenated_documents = df['concatenated'].tolist()
    
    embeddings = np.load("embeddings.npy")
    embeddings = np.array(embeddings).astype('float32')
    
    searcher = HybridSearch(concatenated_documents, embeddings, 0.7)
    
    return df, searcher

def extract_keywords(uploaded_file, tokenizer, model):
    proposal = uploaded_file.getvalue().decode('utf-8')
    
    generator = pipeline(model=model, tokenizer=tokenizer, task="text-generation", device="mps")
    
    messages = [
        {"role": "user", "content": f"What are the research areas mentioned in the following proposal? Your output should only be the research areas, separated by a comma and nothing else. Limit to 5 to 25 keywords at most. Proposal: {proposal}"},
    ]
    
    inference_output = generator(messages, do_sample=False, max_length=4096)
    
    keywords = inference_output[0]["generated_text"][-1]["content"].strip()
    
    return keywords

def create_frequency_chart(professor_frequencies):
    df_freq = pd.DataFrame(professor_frequencies)
    fig = px.bar(
        df_freq,
        x='count',
        y='name',
        orientation='h',
        title='Frequently Appearing Professors',
        labels={'count': 'Number of Relevant Papers', 'name': 'Professor'},
    )
    fig.update_layout(
        showlegend=False,
        xaxis_title="Number of Papers",
        yaxis_title="Professor Name",
        height=max(600, len(professor_frequencies) * 30)
    )
    return fig

def main():
    st.title("AiMap: Discover Researchers at Georgia Tech")
    st.markdown("""
        Find researchers at Georgia Tech based on your research interests or by uploading a research proposal.
        Upload a research proposal or directly enter research area keywords.
    """)
    
    try:
        model = loadEmbeddingModel()
        tokenizer, llm_model = loadLLM()
        df, searcher = initialize_search()
    except Exception as e:
        st.error(f"Error loading the search system: {str(e)}")
        return

    tab1, tab2 = st.tabs(["Keyword Search", "Proposal Upload"])
    
    with tab1:
        st.subheader("Search by Keywords")
        query = st.text_input(
            "Enter your research interests:",
            placeholder="machine learning, computer vision, robotics",
            key="keyword_input"
        )
    
    with tab2:
        st.subheader("Extract Keywords from NSF Proposal")
        uploaded_file = st.file_uploader("Upload a proposal text file", type=['txt'])
        
        if uploaded_file is not None:
            try:
                with st.spinner('Generating keywords...'):
                    extracted_keywords = extract_keywords(uploaded_file, tokenizer, llm_model)
                
                st.success(f"Extracted Keywords: {extracted_keywords}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    if query:
        with st.spinner("Searching..."):
            query_embedding = model.encode(query)
            results = searcher.search(query, query_embedding, top_k=100)
            
            processed_results = []
            repeated_professors = {}
            
            for idx, score, doc in results:
                author_name = df.iloc[idx]["Author"]
                title = df.iloc[idx]["Title"]
                abstract = df.iloc[idx]["Abstract"]
                repeated_professors[author_name] = repeated_professors.get(author_name, 0) + 1
                
                processed_results.append({
                    'author': author_name,
                    'score': float(score),
                    'title': title,
                    'abstract': abstract
                })
            
            professor_frequencies = [
                {'name': author, 'count': count}
                for author, count in sorted(repeated_professors.items(), key=lambda x: x[1], reverse=True)
            ]
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.plotly_chart(
                    create_frequency_chart(professor_frequencies),
                    use_container_width=True
                )
                
                st.metric("Total Matching Professors", len(repeated_professors))
                st.metric("Total Relevant Papers", len(results))
            
            with col2:
                st.subheader("Most Relevant Papers with Scores")
                for rank, result in enumerate(processed_results[:20], 1):
                    st.markdown(f"""
                        **{rank}. {result['author']}** *(Score: {result['score']:.3f})*
                        
                        <div class="small-text">
                        **Title:** {result['title']}
                        
                        **Abstract:** {result['abstract']}
                        </div>
                        
                        ---
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()