## Denser Chat

<img src="demo.gif" width="800" alt="Denser Chat">

Denser Chat is a chatbot that can answer questions from PDFs and webpages. This project is actively developed and maintained by [denser.ai](https://denser.ai). Feel free to contact support@denser.ai if you have feedback or questions.

Main features:

* Extract text and tables from PDFs and webpages.
* Build a chatbot with [denser-retriever](https://github.com/denser-org/denser-retriever)
* Support interactive Streamlit chatbot app with source highlights in PDFs
* Elasticsearch is used for fast retrieval on indexed data, and the elastic keyword enable searching or filtering based on specific terms or criteria within that indexed data.

## Installation

First clone the repository.

```bash
git clone https://github.com/denser-org/denser-chat.git
```

Go to the project directory and start a virtual environment. Make sure your python version is 3.11.

```bash
cd denser-chat
python -m venv .venv
# For Linux/Mac users
source .venv/bin/activate
# For Windows users
.\.venv\Scripts\activate.bat
```

Run the following command to install the required packages.

```bash
pip install -e .
```

Or use this [poetry](https://python-poetry.org/docs/) command

```bash
poetry install
```

## Quick Start

Before building an index, we need to run docker-compose to start Elasticsearch and Milvus services in the background,
which are required for denser-retriever.

```bash
cd denser_chat
docker compose -f docker-compose.yml up -d
```

We run the following command to build a chatbot index. The first argument is the sources file which specify files used to build chatbots. Files can be local PDF files, URL PDFs, or URLs. The second argument is the output directory, and the third argument is the index name.

```bash
python build.py sources.txt output test_index
```

This command will build an index `test_index` via denser-retriever. Next we can start a streamlit app with the following
command. As the app relies on ChatGPT or Claude API, we need to set their keys (one is sufficient) in the environment variables.

```bash
export OPENAI_API_KEY="your-openai-key"
export CLAUDE_API_KEY="your-claude-key"
```

In order to run the app, we need to start a local server to serve the PDFs. We can use the following command to start a server at root directory.

```bash
python -m http.server 8000 
```

Then we can start the streamlit app on a different terminal with the following command.

```bash
cd denser_chat
streamlit run demo.py -- --index_name test_index 
```

for papers
```bash
streamlit run demo.py -- --index_name paper_index
```

Then we can start to ask questions such as "What is in-batch negative sampling ?" or "what parts have stop pins?". We can expect that the chatbot will return the answer with the source highlighted in the PDF.

### License

This project is licensed under the MIT License.