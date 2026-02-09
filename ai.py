import os
from typing import List, Dict, Generator
from openai import OpenAI

class DeepSeekAgent:
    def __init__(self, api_key: str = None):
        """
        Initialize the DeepSeek Agent.
        
        Args:
            api_key: The DeepSeek API key. If None, it will be read from the DEEPSEEK_API_KEY environment variable.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required. Please set DEEPSEEK_API_KEY environment variable or pass it to the constructor.")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def chat(self, user_query: str, context_papers: List[Dict]) -> Generator[str, None, None]:
        """
        Send a query to DeepSeek with context from papers.
        
        Args:
            user_query: The user's question.
            context_papers: A list of dictionaries, each containing 'title' and 'abstract' of a paper.
            
        Yields:
            Chunks of the response text.
        """
        
        # 1. Construct the system prompt and context
        system_prompt = (
            "You are an expert research assistant. You are provided with a list of academic paper abstracts. "
            "Your task is to answer the user's query based ONLY on the provided information. "
            "If the information is not sufficient, state that clearly. "
            "When referencing a paper, mention its title or index. "
            "Please answer in the same language as the user's query (or in Chinese if requested)."
        )
        
        context_text = ""
        for idx, paper in enumerate(context_papers, 1):
            title = paper.get("title", "Unknown Title")
            abstract = paper.get("abstract", "No abstract available.")
            context_text += f"[{idx}] Title: {title}\nAbstract: {abstract}\n\n"
            
        user_message = f"User Query: {user_query}\n\nHere are the candidate papers:\n{context_text}"
        
        # 2. Call the API
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                stream=True
            )
            
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"\n[Error calling DeepSeek API: {str(e)}]"

    def extract_keywords(self, user_query: str) -> str:
        """
        Extract a search keyword from the user query.
        """
        prompt = (
            f"Extract the most important search keyword or phrase (in English) from the following query: '{user_query}'. "
            "Return ONLY the keyword/phrase, without any explanation or quotes."
        )
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return user_query  # Fallback to original query
