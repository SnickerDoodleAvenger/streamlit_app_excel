import os
import pandas as pd
import requests
from dotenv import load_dotenv
import PyPDF2
import openpyxl
from pathlib import Path
import json
import tiktoken
import re
import streamlit as st

# Load environment variables from .env file
load_dotenv()


class DataAnalyzer:
    def __init__(self, excel_path, sop_dir, pdf_dir, model="gpt-4-turbo"):
        """
        Initialize the DataAnalyzer with paths to data sources.

        Args:
            excel_path (str): Path to the Excel file
            sop_dir (str): Directory containing SOP files
            pdf_dir (str): Directory containing PDF files
            model (str): OpenAI model to use
        """
        self.excel_path = excel_path
        self.sop_dir = sop_dir
        self.pdf_dir = pdf_dir
        self.model = model
        self.df = None
        self.sop_content = {}
        self.pdf_content = {}
        self.context = ""

        # Load data sources
        self.load_excel()
        self.load_sops()
        self.load_pdfs()
        self.build_context()

    def load_excel(self):
        """Load and validate the Excel file."""
        try:
            # Try reading with specific header row
            self.df = pd.read_excel(self.excel_path, header=0)

            # Drop any completely empty rows
            self.df = self.df.dropna(how='all')

            # Drop any completely empty columns
            self.df = self.df.dropna(axis=1, how='all')

            # Convert all column names to strings
            self.df.columns = [str(col) for col in self.df.columns]

            print(f"Excel file loaded successfully with {len(self.df)} rows and {len(self.df.columns)} columns.")
            print(f"Columns: {', '.join([str(col) for col in self.df.columns])}")
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            raise

    def load_sops(self):
        """Load SOP files from the specified directory."""
        sop_path = Path(self.sop_dir)
        sop_files = list(sop_path.glob("*.txt")) + list(sop_path.glob("*.pdf"))

        for file in sop_files:
            try:
                if file.suffix.lower() == '.pdf':
                    content = self._extract_text_from_pdf(file)
                else:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()

                self.sop_content[file.name] = content
                print(f"Loaded SOP: {file.name}")
            except Exception as e:
                print(f"Error loading SOP file {file.name}: {e}")

    def load_pdfs(self):
        """Load PDF files from the specified directory."""
        pdf_path = Path(self.pdf_dir)
        pdf_files = list(pdf_path.glob("*.pdf"))

        for file in pdf_files:
            try:
                content = self._extract_text_from_pdf(file)
                self.pdf_content[file.name] = content
                print(f"Loaded PDF: {file.name}")
            except Exception as e:
                print(f"Error loading PDF file {file.name}: {e}")

    def _extract_text_from_pdf(self, file_path):
        """Extract text content from a PDF file."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text += page.extract_text() + "\n\n"
            return text
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    def _summarize_text(self, text, max_words=100):
        """Create a brief summary of text content."""
        words = text.split()
        if len(words) <= max_words:
            return text
        return ' '.join(words[:max_words]) + "..."

    def build_context(self):
        """Build the context from all data sources."""
        # Add Excel data summary
        excel_summary = self._summarize_excel_data()

        # Add SOP summaries
        sop_summary = "SOP Documents:\n"
        for name, content in self.sop_content.items():
            sop_summary += f"- {name}: {self._summarize_text(content, 200)}\n"

        # Build the full context
        self.context = f"""
Excel Data Summary:
{excel_summary}

{sop_summary}
        """
        print(f"Context built with {self._num_tokens(self.context)} tokens.")

    def _summarize_excel_data(self):
        """Create a more comprehensive summary of the Excel data."""
        if self.df is None:
            return "No Excel data loaded."

        summary = f"Dataset contains {len(self.df)} rows and {len(self.df.columns)} columns.\n"
        summary += f"Columns: {', '.join([str(col) for col in self.df.columns])}\n\n"

        # Add sample data (still useful for reference)
        summary += "Sample data (first 5 rows):\n"
        summary += self.df.head(5).to_string() + "\n\n"

        # Add aggregated statistics for key columns
        summary += "=== Aggregated Statistics ===\n"

        # Check if specific columns exist and add relevant summaries
        if 'Region' in self.df.columns and 'Revenue' in self.df.columns:
            region_revenue = self.df.groupby('Region')['Revenue'].sum().reset_index()
            summary += "Revenue by Region:\n"
            summary += region_revenue.to_string() + "\n\n"

        if 'Product_Category' in self.df.columns and 'Revenue' in self.df.columns:
            category_revenue = self.df.groupby('Product_Category')['Revenue'].sum().reset_index()
            summary += "Revenue by Product Category:\n"
            summary += category_revenue.to_string() + "\n\n"

        if 'Channel' in self.df.columns and 'Revenue' in self.df.columns:
            channel_revenue = self.df.groupby('Channel')['Revenue'].sum().reset_index()
            summary += "Revenue by Channel:\n"
            summary += channel_revenue.to_string() + "\n\n"

        # Add basic statistics for numeric columns
        summary += "Basic statistics for numeric columns:\n"
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                stats = self.df[col].describe()
                summary += f"{col}: min={stats['min']}, max={stats['max']}, avg={stats['mean']:.2f}, sum={self.df[col].sum()}\n"

        return summary

    def _num_tokens(self, text):
        """Count the number of tokens in the text."""
        encoding = tiktoken.encoding_for_model(self.model)
        return len(encoding.encode(text))

    def get_relevant_context(self, query, max_tokens=3000):
        """
        Get the most relevant context for a query.

        Args:
            query (str): The user's query
            max_tokens (int): Maximum tokens for context

        Returns:
            str: Relevant context
        """
        # Start with the Excel summary
        context = self._summarize_excel_data()
        tokens_used = self._num_tokens(context)

        # Search SOPs and PDFs for relevant content
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))

        relevant_docs = []
        for name, content in {**self.sop_content, **self.pdf_content}.items():
            doc_keywords = set(re.findall(r'\b\w+\b', content.lower()))
            common_keywords = query_keywords.intersection(doc_keywords)
            score = len(common_keywords) / max(1, len(query_keywords))
            relevant_docs.append((name, content, score))

        # Sort by relevance score
        relevant_docs.sort(key=lambda x: x[2], reverse=True)

        # Add relevant document content until we hit the token limit
        for name, content, score in relevant_docs:
            if score > 0.2:  # Only include if somewhat relevant
                excerpt = self._extract_relevant_excerpt(content, query, 500)
                excerpt_tokens = self._num_tokens(excerpt)

                if tokens_used + excerpt_tokens <= max_tokens:
                    context += f"\nFrom {name}:\n{excerpt}\n"
                    tokens_used += excerpt_tokens
                else:
                    break

        return context

    def _extract_relevant_excerpt(self, text, query, max_words=300):
        """Extract the most relevant excerpt from text based on the query."""
        # Simple keyword matching for now
        query_keywords = set(re.findall(r'\b\w+\b', query.lower()))

        # Split text into paragraphs and score each
        paragraphs = text.split('\n\n')
        scored_paragraphs = []

        for para in paragraphs:
            if len(para.strip()) == 0:
                continue

            para_keywords = set(re.findall(r'\b\w+\b', para.lower()))
            common_keywords = query_keywords.intersection(para_keywords)
            score = len(common_keywords)
            scored_paragraphs.append((para, score))

        # Sort by relevance score
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)

        # Take the most relevant paragraphs up to max_words
        result = ""
        word_count = 0

        for para, score in scored_paragraphs:
            if score > 0:  # Only include if there's a keyword match
                para_words = len(para.split())
                if word_count + para_words <= max_words:
                    result += para + "\n\n"
                    word_count += para_words
                else:
                    remaining_words = max_words - word_count
                    result += ' '.join(para.split()[:remaining_words]) + "..."
                    break

        return result

    def analyze_data(self, specific_analysis=None):
        """
        Provide a general analysis of the data using OpenAI.

        Args:
            specific_analysis (str): Optional specific aspect to analyze

        Returns:
            str: Analysis results
        """
        # Prepare the prompt
        if specific_analysis:
            prompt = f"""
Analyze the following Excel data with focus on {specific_analysis}.
Provide insights, trends, and notable observations.

{self._summarize_excel_data()}
            """
        else:
            prompt = f"""
Analyze the following Excel data.
Provide a comprehensive narrative that includes:
1. Key insights and patterns
2. Notable trends
3. Anomalies or outliers
4. Potential business implications

{self._summarize_excel_data()}
            """

        # Call OpenAI API directly with requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You are a data analyst providing clear, insightful analysis of Excel data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 1000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            response_data = response.json()

            # Track token usage if session state has token tracker
            if hasattr(st.session_state, 'token_tracker'):
                st.session_state.token_tracker.track_api_call(
                    prompt=prompt,
                    response=response_data,
                    model=self.model
                )

            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def answer_question(self, question):
        """
        Answer a question about the data using OpenAI.

        Args:
            question (str): The user's question about the data

        Returns:
            str: The answer
        """
        # Get relevant context for this specific question
        relevant_context = self.get_relevant_context(question)

        # Prepare the prompt
        prompt = f"""
Based on the following data and documentation, please answer this question:
"{question}"

Context information:
{relevant_context}

Please provide a clear, accurate answer based only on the information provided. 
If the answer cannot be determined from the information, please state that clearly.
        """

        # Call OpenAI API directly with requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system",
                 "content": "You are a data analyst who provides accurate, helpful answers based on the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            response_data = response.json()

            # Track token usage if session state has token tracker
            if hasattr(st.session_state, 'token_tracker'):
                st.session_state.token_tracker.track_api_call(
                    prompt=prompt,
                    response=response_data,
                    model=self.model
                )

            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def generate_report(self, report_type):
        """
        Generate a specific type of report on the data.

        Args:
            report_type (str): Type of report to generate (summary, detailed, executive)

        Returns:
            str: The generated report
        """
        # Map report types to specific prompts
        report_prompts = {
            "summary": "Provide a brief summary of the key insights from this data in 3-5 bullet points.",
            "detailed": "Generate a detailed analytical report with sections for methodology, findings, and recommendations.",
            "executive": "Create an executive summary focusing on business implications and actionable insights."
        }

        prompt = report_prompts.get(report_type.lower(),
                                    "Provide a comprehensive analysis of this data.")

        # Prepare the full prompt with data
        full_prompt = f"""
{prompt}

Data context:
{self._summarize_excel_data()}

Relevant SOPs and documentation:
{self._summarize_sop_content()}
        """

        # Call OpenAI API directly with requests
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are a business analyst creating professional reports from data."},
                {"role": "user", "content": full_prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 1500
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            response.raise_for_status()  # Raise exception for HTTP errors
            response_data = response.json()

            # Track token usage if session state has token tracker
            if hasattr(st.session_state, 'token_tracker'):
                st.session_state.token_tracker.track_api_call(
                    prompt=full_prompt,
                    response=response_data,
                    model=self.model
                )

            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            raise

    def _summarize_sop_content(self):
        """Create a summary of SOP content."""
        summary = ""
        for name, content in self.sop_content.items():
            summary += f"From {name}:\n{self._summarize_text(content, 200)}\n\n"
        return summary
