from __future__ import annotations
from pathlib import Path
from llama_cpp import Llama


class LLMGenerator:
    """
    Component 5: LLM generation using TinyLlama-1.1B-Chat-v0.3.
    Takes augmented prompts and generates natural language responses.
    """
    
    def __init__(self, model_path: Path):
        print(f"Loading LLM from {model_path}...")
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=2048,
            verbose=False,
        )
        print("LLM ready.")
    
    def generate(self, prompt: str, max_tokens: int = 256) -> str:
        """
        Generate a response from the LLM given a prompt.
        
        Args:
            prompt: The augmented prompt containing query + context
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Use chat format for TinyLlama
        chat_prompt = f"<|user|>\n{prompt}<|assistant|>\n"
        
        response = self.llm(
            chat_prompt,
            max_tokens=max_tokens,
            temperature=0.7,
            stop=["<|user|>"],  # Stop if user tag appears (new query)
            echo=False,
        )
        
        text = response["choices"][0]["text"].strip()
        
        # If empty, try without chat format as fallback
        if not text:
            response = self.llm(
                prompt,
                max_tokens=max_tokens,
                temperature=0.7,
                stop=[],
                echo=False,
            )
            text = response["choices"][0]["text"].strip()
        
        return text if text else "I couldn't generate a response. Please try rephrasing your question."
