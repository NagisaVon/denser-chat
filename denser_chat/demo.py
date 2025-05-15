from openai import OpenAI
import streamlit as st
import time
import os
from denser_retriever.keyword import (
    ElasticKeywordSearch,
    create_elasticsearch_client,
)
from denser_retriever.retriever import DenserRetriever
import json
import logging
import anthropic
import argparse
from urllib.parse import urlencode, quote
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()
# Define available models
MODEL_OPTIONS = {
    "GPT-4.1": "gpt-4.1",
    "GPT-o4-mini": "o4-mini",
}
context_window = 128000
# Get API keys from environment variables with optional default values
openai_api_key = os.getenv('OPENAI_API_KEY')
claude_api_key = os.getenv('CLAUDE_API_KEY')

# Check if API keys are set
if not (openai_api_key or claude_api_key):
    raise ValueError("Neither OPENAI_API_KEY nor CLAUDE_API_KEY environment variables is set")

openai_client = OpenAI(api_key=openai_api_key)
claude_client = anthropic.Client(api_key=claude_api_key)
history_turns = 5

prompt_default = "### Instructions:\n" \
                 "You are a professional AI assistant. The following context consists of an ordered list of sources. " \
                 "If you can find answers from the context, use the context to provide a response. " \
                 "You must cite passages in square brackets [X] where X is the passage number (do not include passage word, only digit numbers)." \
                 "If you cannot find the answer from the sources, use your knowledge to come up a reasonable answer. " \
                 "If the query asks to summarize the file or uploaded file, provide a summarization based on the provided sources. " \
                 "If the conversation involves casual talk or greetings, rely on your knowledge for an appropriate response. "


def create_viewer_url_by_passage(passage):
    """Create a URL to open PDF.js viewer with annotation highlighting."""
    base_url = "https://laughing-fortnight-wp7g4gww752grqv-8000.app.github.dev/viewer.html"

    try:
        ann_list = json.loads(passage[0].metadata.get('annotations', '[]'))
        pdf_url = passage[0].metadata.get('source', None)
        if not pdf_url or not ann_list:
            return None

        # Convert each annotation to include page information
        viewer_annotations = []
        for ann in ann_list:
            viewer_annotations.append({
                'x': ann.get('x', 0),
                'y': ann.get('y', 0),
                'width': ann.get('width', 0),
                'height': ann.get('height', 0),
                'page': ann.get('page', 0)  # Include the page number for each annotation
            })

        # Create a single URL with all annotations
        params = {
            'file': pdf_url,
            'annotations': json.dumps(viewer_annotations),
            'pageNumber': viewer_annotations[0]['page'] + 1  # Start with first annotated page
        }
        return f"{base_url}?{urlencode(params, quote_via=quote)}"

    except (json.JSONDecodeError, AttributeError) as e:
        print(f"Error in create_viewer_url_by_passage: {e}")
        return None


def post_process_html(full_response: str, passages: list) -> str:
    """Similar to post_process but outputs HTML links that open in new tab."""
    import re

    def replace_citation(match):
        num = int(match.group(1)) - 1
        if num < len(passages):
            source = passages[num][0].metadata.get('source', '')
            # Open a new tab for all URLs except PDFs
            if source.startswith(('http://', 'https://')) and not source.endswith('.pdf'):
                return f'<a href="{source}" target="_blank">[{num + 1}]</a>'
            else:
                viewer_url = create_viewer_url_by_passage(passages[num])
                if viewer_url:
                    return f'<a href="{viewer_url}" target="_blank">[{num + 1}]</a>'
                else:
                    return f'[{num + 1}]'
        return match.group(0)

    processed_text = re.sub(r'\[(\d+)\]', replace_citation, full_response)
    return processed_text


def stream_response(selected_model, messages, passages):

    """
    Stream assistant response based on the selected model.
    
    Parameters:
    - selected_model_name: The name of the model as a key in MODEL_OPTIONS from user.
    - messages: List of messages for the chat input.
    """

    # Generate and display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        print(f"Using OpenAI {selected_model} model")
        messages.insert(0, {"role": "system", "content": "You are a helpful assistant. You should present the answer in the structure that best explaining the subject. When not sure, follow this pattern: First answer the question straight forward, then provide the detailed reasoning leading towards the answer. You avoid being robotic, overly polite, or saccharine. Be playful and use some humor when appropriate. When mentioning technical term that I likely don’t know, add a short explaination using simple term, example or analogy, but don’t force an analogy if there isn’t a good one. For any technical questions regarding API, gcloud platform, library, language feature as such, always conduct a search and reference the latest documentation. Reply in Simplified Chinese, use the original English words for technical term."})
        for response in openai_client.chat.completions.create(
                model=selected_model,
                messages=messages,
                stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        
    # Update session state
    print(f"### messages\n: {messages}\n")
    print(f"### full_response\n: {full_response}\n")
    processed_response = post_process_html(full_response, passages)
    print(f"### processed_response\n: {processed_response}\n")
    st.session_state.messages.append({"role": "assistant", "content": processed_response})
    st.session_state.passages = passages

    st.rerun()  # Add rerun here to show links immediately


def main(args):
    st.set_page_config(layout="wide")

    global retriever
    retriever = DenserRetriever(
        index_name=args.index_name,
        keyword_search=ElasticKeywordSearch(
            top_k=100,
            es_connection=create_elasticsearch_client(url="http://localhost:9200",
                                                      username="elastic",
                                                      password="",
                                                      ),
            drop_old=False,
            analysis="default"  # default or ik
        ),
        vector_db=None,
        reranker=None,
        embeddings=None,
        gradient_boost=None,
        search_fields=["annotations:keyword"],
    )

    # Initialize session states
    if 'selected_model' not in st.session_state:
        st.session_state.selected_model = list(MODEL_OPTIONS.keys())[0]  # Default model
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "passages" not in st.session_state:
        st.session_state.passages = []

    # Create header with title and model selector
    st.title("Denser Chat Demo")
    selected_model_name = st.selectbox(
        "Select Model",
        options=list(MODEL_OPTIONS.keys()),
        key="model_selector",
        index=list(MODEL_OPTIONS.keys()).index(st.session_state.selected_model)
    )
    st.session_state.selected_model = selected_model_name

    st.caption(
        "Try question \"What is example domain?\", \"What is in-batch negative sampling?\" or \"what parts have stop pins?\"")
    st.divider()

    for i in range(len(st.session_state.messages)):
        with st.chat_message(st.session_state.messages[i]["role"]):
            st.markdown(st.session_state.messages[i]["content"], unsafe_allow_html=True)

    # Handle user input
    query = st.chat_input("Please input your question")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        start_time = time.time()
        passages = retriever.retrieve(query, 5, {})
        retrieve_time_sec = time.time() - start_time
        st.write(f"Retrieve time: {retrieve_time_sec:.3f} sec.")

        # Process chat completion
        prompt = prompt_default + f"### Query:\n{query}\n"
        if len(passages) > 0:
            prompt += "\n### Context:\n"
            for i, passage in enumerate(passages):
                prompt += f"#### Passage {i + 1}:\n{passage[0].page_content}\n"

        if args.language == "en":
            context_limit = 4 * context_window
        else:
            context_limit = context_window
        prompt = prompt[:context_limit] + "### Response:"

        # Prepare messages for chat completion
        messages = st.session_state.messages[-history_turns * 2:]
        messages.append({"role": "user", "content": prompt})

        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": query})

        stream_response(MODEL_OPTIONS[selected_model_name], messages, passages)


def parse_args():
    parser = argparse.ArgumentParser(description='Denser Chat Demo')
    parser.add_argument('--index_name', type=str, default=None,
                        help='Name of the Elasticsearch index to use')
    parser.add_argument('--language', type=str, default='en',
                        help='Language setting for context window (en or ch, default: en)')
    parser.add_argument('--static_dir', type=str, default='static',
                        help='Directory where PDF.js and PDFs are served from')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
