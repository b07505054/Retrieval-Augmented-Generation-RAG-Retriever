# Retrieval-Augmented Generation (RAG) Assignment

This repository contains the implementation of a Retrieval-Augmented Generation (RAG) system focused on the retrieval component, as part of an assignment for CSE256. The project uses the [QMSum dataset](https://github.com/Yale-LILY/QMSum) to build a retriever that finds relevant documents from meeting transcripts given a user query. The implementation leverages **LangChain**, **HuggingFace models**, and **FAISS** for efficient neural retrieval.

The workflow is based on the RAG pipeline, as illustrated below (diagram sourced from Hugging Face):

![RAG Workflow](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/RAG_workflow.png)

## Project Overview

The goal of this project is to build a retriever that:
- Processes long meeting transcripts from the QMSum dataset.
- Splits documents into semantically meaningful chunks.
- Embeds the chunks using a pretrained text encoder (`thenlper/gte-small`).
- Stores embeddings in a FAISS vector database for efficient similarity search.
- Retrieves the top-k most relevant documents for a given user query using a Bi-Encoder.
- Re-ranks the retrieved documents using a simplified version of the ColBERT approach.

This project focuses solely on the **Retriever** component (Step 1 in the RAG workflow diagram) and does not cover the Reader or generation steps.

## Dataset

The project uses the [QMSum dataset](https://github.com/Yale-LILY/QMSum), which includes:
- **meetings.tsv**: Contains 230 meeting transcripts with document IDs and content.
- **questions_answers.tsv**: Contains question-answer pairs with corresponding document IDs for evaluation.

**Note**: Due to GitHub's file size limits, the dataset files are hosted on Google Drive. You can download them from the following link:
- [Download Dataset](https://drive.google.com/drive/folders/1I2HDua6Gh3tojE2UIRnfbTqQViCcW73o?usp=sharing)

## Requirements

To run this project, you need Python 3.8+ and the following dependencies:

```bash
pip install -r requirements.txt
