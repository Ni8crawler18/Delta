"""
app.py - Streamlit Dashboard for Truth Tell

Real-time misinformation detection dashboard for live broadcasts.
"""

import streamlit as st
import sys
import os
import time
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import modules
from model.01_conversion import YouTubeConverter, preprocess_text
from model.02_claims import ClaimsDetector
from model.04_summarize import FactSummarizer

# Constants
FACT_CHECK_API_URL = "http://localhost:5000/api/factcheck"
SAMPLE_VIDEO_PATH = "../assets/news_sample.mp4"

# Initialize models
@st.cache_resource
def load_models():
    converter = YouTubeConverter(model_size="base")
    detector = ClaimsDetector()
    summarizer = FactSummarizer()
    return converter, detector, summarizer

converter, detector, summarizer = load_models()

# Page configuration
st.set_page_config(
    page_title="Truth Tell - Misinformation Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stProgress .st-eb {height: 25px;}
    .verdict-card {padding: 20px; border-radius: 5px; margin-bottom: 10px;}
    .verdict-false {background-color: rgba(255, 0, 0, 0.1); border-left: 5px solid red;}
    .verdict-true {background-color: rgba(0, 255, 0, 0.1); border-left: 5px solid green;}
    .verdict-mixed {background-color: rgba(255, 165, 0, 0.1); border-left: 5px solid orange;}
    .verdict-unverified {background-color: rgba(128, 128, 128, 0.1); border-left: 5px solid gray;}
    .highlight {background-color: yellow; padding: 3px 0;}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Truth tell")
st.sidebar.markdown("## Misinformation Detection")

# Input options
input_type = st.sidebar.radio("Select Input Source", ["YouTube URL", "Sample Video", "Text Input"])

if input_type == "YouTube URL":
    youtube_url = st.sidebar.text_input("Enter YouTube URL")
    process_button = st.sidebar.button("Process Video")
elif input_type == "Sample Video":
    st.sidebar.markdown(f"Using sample video: {SAMPLE_VIDEO_PATH}")
    process_button = st.sidebar.button("Process Sample Video")
else:
    text_input = st.sidebar.text_area("Enter text to analyze", height=200)
    process_button = st.sidebar.button("Analyze Text")

# Advanced options
with st.sidebar.expander("Advanced Options"):
    threshold = st.slider("Misinformation Threshold", 0.5, 0.9, 0.7, 0.05)
    real_time = st.checkbox("Real-time Analysis", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Project Delta")

# Main content
st.title("🔍 Truth Tell Dashboard")
st.markdown("Real-time misinformation detection in live broadcasting")

# Process flow
if process_button:
    # Container for progress
    progress_container = st.container()
    
    with progress_container:
        if input_type in ["YouTube URL", "Sample Video"]:
            st.subheader("Processing Video")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Step 1: Download and convert video
            status_text.text("Step 1/4: Downloading and converting video...")
            
            if input_type == "YouTube URL":
                try:
                    result = converter.process_youtube_url(youtube_url)
                    text = result['punctuated_text']
                except Exception as e:
                    st.error(f"Error processing YouTube URL: {str(e)}")
                    st.stop()
            else:
                # Use sample video
                sample_path = os.path.abspath(SAMPLE_VIDEO_PATH)
                if not os.path.exists(sample_path):
                    st.error(f"Sample video not found at {sample_path}")
                    st.stop()
                
                try:
                    result = converter.transcribe_audio(sample_path)
                    text = converter.restore_punctuation(result['text'])
                except Exception as e:
                    st.error(f"Error processing sample video: {str(e)}")
                    st.stop()
            
            progress_bar.progress(25)
        else:
            # Text input
            st.subheader("Processing Text Input")
            progress_bar = st.progress(25)
            status_text = st.empty()
            text = text_input
        
        # Step 2: Detect claims
        status_text.text("Step 2/4: Detecting and classifying claims...")
        preprocessed_text = preprocess_text(text)
        claims_result = detector.process_text(preprocessed_text, threshold=threshold)
        progress_bar.progress(50)
        
        # Step 3: Fact check claims
        status_text.text("Step 3/4: Fact checking claims...")
        
        try:
            response = requests.post(
                FACT_CHECK_API_URL,
                json={'claims': claims_result['claims']},
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                fact_check_results = response.json()['results']
            else:
                st.warning(f"Fact check API returned status {response.status_code}. Using mock data.")
                # Generate mock results if API fails
                fact_check_results = []
                for claim in claims_result['claims']:
                    fact_check_results.append({
                        'claim': claim,
                        'google_results': [],
                        'external_results': [],
                        'verdict': {
                            'rating': 'Unverified',
                            'confidence': 0.5,
                            'explanation': 'Unable to fact check through API'
                        }
                    })
        except Exception as e:
            st.warning(f"Error connecting to fact check API: {str(e)}. Using mock data.")
            # Generate mock results if API fails
            fact_check_results = []
            for claim in claims_result['claims']:
                fact_check_results.append({
                    'claim': claim,
                    'google_results': [],
                    'external_results': [],
                    'verdict': {
                        'rating': 'Unverified',
                        'confidence': 0.5,
                        'explanation': 'Unable to fact check through API'
                    }
                })
        
        progress_bar.progress(75)
        
        # Step 4: Generate summary
        status_text.text("Step 4/4: Generating summary...")
        summary_result = summarizer.generate_overall_summary(fact_check_results, full_text=text)
        progress_bar.progress(100)
        status_text.text("Analysis complete!")
        time.sleep(1)
        
        # Clear progress indicators
        progress_container.empty()
    
    # Display results in tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Claims Analysis", "Fact Check Results", "Full Text"])
    
    # Tab 1: Summary
    with tab1:
        st.header("Analysis Summary")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                "Misinformation Score",
                f"{summary_result['misinformation_score'] * 100:.1f}%",
                delta=None,
                delta_color="inverse"
            )
            
            # Color-coded gauge chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=summary_result['misinformation_score'] * 100,
                title={'text': "Misinformation Level"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgreen"},
                        {'range': [33, 66], 'color': "khaki"},
                        {'range': [66, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': summary_result['misinformation_score'] * 100
                    }
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Verdicts distribution
            st.subheader("Claims Verdicts")
            verdicts = summary_result['verdicts']
            
            # Prepare data for pie chart
            labels = []
            values = []
            colors = []
            
            for verdict, count in verdicts.items():
                if count > 0:
                    labels.append(verdict)
                    values.append(count)
                    if verdict in ['False', 'Likely False']:
                        colors.append('red')
                    elif verdict in ['True', 'Likely True']:
                        colors.append('green')
                    elif verdict == 'Mixed':
                        colors.append('orange')
                    else:
                        colors.append('gray')
            
            fig = px.pie(
                names=labels,
                values=values,
                color_discrete_sequence=colors,
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col3:
            st.subheader("Analysis")
            st.markdown(summary_result['misinformation_analysis'])
            
            # Claims count
            st.markdown(f"""
            - **Total Claims**: {claims_result['potential_claims']}
            - **Potential Misinformation**: {claims_result['misinformation_claims']}
            - **False Claims**: {verdicts['False'] + verdicts['Likely False']}
            - **True Claims**: {verdicts['True'] + verdicts['Likely True']}
            """)
        
        # Important claims
        st.subheader("Key Claims Analyzed")
        for i, claim in enumerate(summary_result['important_claims']):
            verdict = claim['verdict']
            
            if verdict in ['False', 'Likely False']:
                verdict_class = "verdict-false"
            elif verdict in ['True', 'Likely True']:
                verdict_class = "verdict-true"
            elif verdict == 'Mixed':
                verdict_class = "verdict-mixed"
            else:
                verdict_class = "verdict-unverified"
            
            st.markdown(f"""
            <div class="verdict-card {verdict_class}">
                <h4>Claim {i+1}: {verdict}</h4>
                <p><strong>Statement:</strong> "{claim['claim']}"</p>
                <p><strong>Analysis:</strong> {claim['summary']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Content summary
        st.subheader("Content Summary")
        st.markdown(summary_result['content_summary'])
        
    # Tab 2: Claims Analysis
    with tab2:
        st.header("Claims Detection & Analysis")
        
        if claims_result['claims']:
            # Filter to misinformation claims
            st.markdown(f"### Detected {len(claims_result['claims'])} potential claims")
            
            # Create dataframe for easier visualization
            claims_data = []
            for claim in claims_result['claims']:
                claims_data.append({
                    'text': claim['text'],
                    'misinformation_score': claim['misinformation_score'],
                    'factual_score': claim['factual_score']
                })
            
            claims_df = pd.DataFrame(claims_data)
            
            # Sort by misinformation score (descending)
            claims_df = claims_df.sort_values('misinformation_score', ascending=False)
            
            # Display bar chart
            fig = px.bar(
                claims_df,
                x='misinformation_score',
                y='text',
                orientation='h',
                title='Claims Ranked by Misinformation Score',
                color='misinformation_score',
                color_continuous_scale=['green', 'yellow', 'red'],
                labels={'misinformation_score': 'Misinformation Score', 'text': 'Claim'}
            )
            fig.update_layout(height=400 + 30 * len(claims_df))
            st.plotly_chart(fig, use_container_width=True)
            
            # Display table
            st.markdown("### Claims Details")
            styled_df = claims_df.style.background_gradient(
                subset=['misinformation_score'],
                cmap='RdYlGn_r'
            ).background_gradient(
                subset=['factual_score'],
                cmap='RdYlGn'
            ).format({
                'misinformation_score': '{:.2%}',
                'factual_score': '{:.2%}'
            })
            st.dataframe(styled_df, use_container_width=True)
        else:
            st.info("No claims detected in the content.")
    
    # Tab 3: Fact Check Results
    with tab3:
        st.header("Fact Check Results")
        
        if fact_check_results:
            for i, result in enumerate(fact_check_results):
                with st.expander(f"Claim {i+1}: {result['claim']['text'][:100]}..."):
                    # Verdict
                    verdict = result['verdict']
                    
                    if verdict['rating'] in ['False', 'Likely False']:
                        verdict_color = "red"
                    elif verdict['rating'] in ['True', 'Likely True']:
                        verdict_color = "green"
                    elif verdict['rating'] == 'Mixed':
                        verdict_color = "orange"
                    else:
                        verdict_color = "gray"
                    
                    st.markdown(f"""
                    <div style="padding: 10px; border-radius: 5px; background-color: {verdict_color}25; border-left: 5px solid {verdict_color};">
                        <h4 style="color: {verdict_color};">{verdict['rating']} (Confidence: {verdict['confidence']:.2f})</h4>
                        <p>{verdict['explanation']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Google Fact Check API results
                    if result['google_results']:
                        st.subheader("Google Fact Check API Results")
                        for i, check in enumerate(result['google_results']):
                            st.markdown(f"""
                            **Fact Check #{i+1}**  
                            Source: {check.get('publisher', {}).get('name', 'Unknown')}  
                            Rating: {check.get('textualRating', 'Not provided')}  
                            URL: [{check.get('url', '#')}]({check.get('url', '#')})
                            """)
                    
                    # External fact check results
                    if result['external_results']:
                        st.subheader("External Fact Check Results")
                        for i, check in enumerate(result['external_results']):
                            st.markdown(f"""
                            **Source #{i+1}**: {check.get('source', 'Unknown')}  
                            Title: {check.get('title', 'Not provided')}  
                            Rating: {check.get('rating', 'Not provided')}  
                            URL: [{check.get('link', '#')}]({check.get('link', '#')})
                            """)
        else:
            st.info("No fact check results available.")
    
    # Tab 4: Full Text
    with tab4:
        st.header("Original Text")
        
        # Highlight claims in the original text
        if 'text' in locals() and claims_result['claims']:
            highlighted_text = text
            
            # Highlight each claim
            for claim in claims_result['claims']:
                claim_text = claim['text']
                if claim_text in highlighted_text:
                    verdict_class = "false" if claim['misinformation_score'] > 0.5 else "true"
                    highlighted_text = highlighted_text.replace(
                        claim_text,
                        f'<span class="highlight {verdict_class}">{claim_text}</span>'
                    )
            
            st.markdown(highlighted_text, unsafe_allow_html=True)
        elif 'text' in locals():
            st.markdown(text)
        else:
            st.info("No text available.")

else:
    # Initial state or when no video is processed
    st.markdown("""
    ## Welcome to Truth Tell
    
    This dashboard helps detect and analyze misinformation in live broadcasts or any text content.
    
    **How it works:**
    1. Provide a YouTube URL, use our sample video, or enter text directly
    2. The system transcribes audio to text (for video sources)
    3. AI models identify potential claims and misinformation
    4. Claims are fact-checked against trusted sources
    5. A detailed analysis and summary are provided
    
    **Get started by:**
    - Selecting an input source in the sidebar
    - Adjusting any advanced settings if needed
    - Clicking the "Process" button
    """)
    
    # Example visualizations for initial state
    st.markdown("### Sample Dashboard Elements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Dummy misinformation gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            title={'text': "Example Misinformation Level"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 33], 'color': "lightgreen"},
                    {'range': [33, 66], 'color': "khaki"},
                    {'range': [66, 100], 'color': "lightcoral"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Dummy pie chart
        fig = px.pie(
            names=["True", "False", "Mixed", "Unverified"],
            values=[70, 10, 15, 5],
            color_discrete_sequence=['green', 'red', 'orange', 'gray'],
            hole=0.4,
            title="Example Verdicts Distribution"
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)