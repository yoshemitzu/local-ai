#!/usr/bin/env python3
"""
Quick LLM - Lightweight Local LLM Utility
A simple, fast interface for local LLM interactions with structured output support.
"""

import json
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import time
from dataclasses import dataclass
from enum import Enum

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("Warning: llama-cpp-python not available. Install with: pip install llama-cpp-python")

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OutputFormat(Enum):
    """Supported output formats"""
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    LIST = "list"
    DICT = "dict"

@dataclass
class LLMConfig:
    """Configuration for LLM model"""
    model_path: str = "WesPro/Mistral-Small-3.1-24B-Instruct-2503-HF-Q6_K-GGUF"
    model_file: str = "mistral-small-3.1-24b-instruct-2503-hf-q6_k.gguf"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    echo: bool = False
    verbose: bool = False

class QuickLLM:
    """Lightweight LLM utility for quick structured outputs"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the LLM model"""
        if not LLAMA_AVAILABLE:
            raise ImportError("llama-cpp-python is required. Install with: pip install llama-cpp-python")
        
        try:
            logger.info(f"Loading model: {self.config.model_path}")
            self.llm = Llama.from_pretrained(
                repo_id=self.config.model_path,
                filename=self.config.model_file,
                verbose=self.config.verbose
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _format_prompt(self, prompt: str, output_format: OutputFormat, schema: Optional[Dict] = None) -> str:
        """Format prompt with output format instructions"""
        format_instructions = {
            OutputFormat.JSON: "Respond with valid JSON only. Do not include any other text.",
            OutputFormat.YAML: "Respond with valid YAML only. Do not include any other text.",
            OutputFormat.LIST: "Respond with a simple list, one item per line.",
            OutputFormat.DICT: "Respond with key-value pairs in a structured format.",
            OutputFormat.TEXT: "Respond naturally with clear, concise text."
        }
        
        formatted_prompt = f"{prompt}\n\n{format_instructions[output_format]}"
        
        if schema and output_format in [OutputFormat.JSON, OutputFormat.YAML]:
            schema_str = json.dumps(schema, indent=2) if output_format == OutputFormat.JSON else yaml.dump(schema)
            formatted_prompt += f"\n\nExpected structure:\n{schema_str}"
        
        return formatted_prompt
    
    def _parse_output(self, output: str, output_format: OutputFormat) -> Any:
        """Parse LLM output based on format"""
        try:
            if output_format == OutputFormat.JSON:
                return json.loads(output)
            elif output_format == OutputFormat.YAML and YAML_AVAILABLE:
                return yaml.safe_load(output)
            elif output_format == OutputFormat.LIST:
                return [line.strip() for line in output.strip().split('\n') if line.strip()]
            elif output_format == OutputFormat.DICT:
                # Simple key-value parsing
                result = {}
                for line in output.strip().split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        result[key.strip()] = value.strip()
                return result
            else:
                return output.strip()
        except Exception as e:
            logger.warning(f"Failed to parse output as {output_format.value}: {e}")
            return output.strip()
    
    def quick_call(self, 
                   prompt: str, 
                   output_format: OutputFormat = OutputFormat.TEXT,
                   schema: Optional[Dict] = None,
                   max_tokens: Optional[int] = None) -> Any:
        """
        Quick LLM call with structured output
        
        Args:
            prompt: Input prompt
            output_format: Desired output format
            schema: Optional schema for structured outputs
            max_tokens: Override default max tokens
            
        Returns:
            Parsed output in the requested format
        """
        if not self.llm:
            raise RuntimeError("LLM not initialized")
        
        formatted_prompt = self._format_prompt(prompt, output_format, schema)
        tokens = max_tokens or self.config.max_tokens
        
        try:
            start_time = time.time()
            response = self.llm(
                formatted_prompt,
                max_tokens=tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                echo=self.config.echo
            )
            elapsed = time.time() - start_time
            
            # Extract the generated text
            if isinstance(response, dict) and 'choices' in response:
                output_text = response['choices'][0]['text']
            else:
                output_text = str(response)
            
            parsed_output = self._parse_output(output_text, output_format)
            
            if self.config.verbose:
                logger.info(f"Generated {len(output_text)} characters in {elapsed:.2f}s")
            
            return parsed_output
            
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def extract_info(self, text: str, fields: List[str]) -> Dict[str, Any]:
        """
        Extract specific information from text
        
        Args:
            text: Input text to analyze
            fields: List of fields to extract
            
        Returns:
            Dictionary with extracted information
        """
        schema = {field: "string" for field in fields}
        prompt = f"Extract the following information from this text:\n{text}\n\nFields to extract: {', '.join(fields)}"
        
        return self.quick_call(prompt, OutputFormat.JSON, schema)
    
    def summarize(self, text: str, max_length: int = 100) -> str:
        """Generate a summary of the given text"""
        prompt = f"Summarize this text in {max_length} characters or less:\n\n{text}"
        return self.quick_call(prompt, OutputFormat.TEXT)
    
    def classify(self, text: str, categories: List[str]) -> str:
        """Classify text into one of the given categories"""
        prompt = f"Classify this text into one of these categories: {', '.join(categories)}\n\nText: {text}"
        return self.quick_call(prompt, OutputFormat.TEXT)
    
    def translate(self, text: str, target_language: str) -> str:
        """Translate text to target language"""
        prompt = f"Translate this text to {target_language}:\n\n{text}"
        return self.quick_call(prompt, OutputFormat.TEXT)

def quick_llm_call(prompt: str, 
                   output_format: str = "text",
                   **kwargs) -> Any:
    """
    Convenience function for quick LLM calls
    
    Args:
        prompt: Input prompt
        output_format: "text", "json", "yaml", "list", or "dict"
        **kwargs: Additional arguments for QuickLLM
    
    Returns:
        LLM response in requested format
    """
    try:
        format_enum = OutputFormat(output_format.lower())
    except ValueError:
        raise ValueError(f"Invalid output format: {output_format}. Use: text, json, yaml, list, dict")
    
    llm = QuickLLM()
    return llm.quick_call(prompt, format_enum, **kwargs)

# Example usage and testing
if __name__ == "__main__":
    # Initialize the LLM
    llm = QuickLLM()
    
    # Example 1: Simple text generation
    print("=== Text Generation ===")
    response = llm.quick_call("Write a haiku about programming")
    print(response)
    print()
    
    # Example 2: JSON output
    print("=== JSON Output ===")
    schema = {
        "name": "string",
        "age": "number",
        "skills": ["string"]
    }
    response = llm.quick_call(
        "Create a developer profile for John, a 28-year-old Python developer",
        OutputFormat.JSON,
        schema
    )
    print(json.dumps(response, indent=2))
    print()
    
    # Example 3: Information extraction
    print("=== Information Extraction ===")
    text = "Alice is a 25-year-old software engineer who works at Google. She specializes in machine learning and Python."
    extracted = llm.extract_info(text, ["name", "age", "job", "company", "skills"])
    print(json.dumps(extracted, indent=2))
    print()
    
    # Example 4: Quick convenience function
    print("=== Quick Call ===")
    result = quick_llm_call("List 3 benefits of local LLMs", "list")
    print(result) 