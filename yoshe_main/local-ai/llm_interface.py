#!/usr/bin/env python3
"""
LLM Interface - Standardized Communication Protocol
Provides unified interfaces for different LLM types and local models.
"""

import json
import time
import asyncio
import subprocess
import requests
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class LLMProvider(Enum):
    """Supported LLM providers"""
    LOCAL_MISTRAL = "local_mistral"
    GEMINI_CLI = "gemini_cli"
    CLAUDE_CLI = "claude_cli"
    GPT_CLI = "gpt_cli"
    OLLAMA = "ollama"
    CUSTOM = "custom"

@dataclass
class LLMResponse:
    """Standardized LLM response"""
    content: str
    provider: LLMProvider
    model_name: str
    tokens_used: Optional[int] = None
    response_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class LLMConfig:
    """Configuration for LLM instances"""
    provider: LLMProvider
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: int = 512
    temperature: float = 0.7
    timeout: int = 30
    working_directory: Optional[str] = None
    command_template: Optional[str] = None

class BaseLLMInterface(ABC):
    """Abstract base class for LLM interfaces"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.is_available = False
        self._test_connection()
    
    @abstractmethod
    def _test_connection(self) -> bool:
        """Test if the LLM is available"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the LLM process is running"""
        pass
    
    @abstractmethod
    def start(self) -> bool:
        """Start the LLM if not running"""
        pass
    
    @abstractmethod
    def stop(self) -> bool:
        """Stop the LLM process"""
        pass

class LocalMistralInterface(BaseLLMInterface):
    """Interface for local Mistral model via llama-cpp"""
    
    def __init__(self, config: LLMConfig):
        try:
            from llama_cpp import Llama
            self.llama_available = True
        except ImportError:
            self.llama_available = False
            logger.error("llama-cpp-python not available")
        
        super().__init__(config)
        self.llm = None
    
    def _test_connection(self) -> bool:
        if not self.llama_available:
            return False
        
        try:
            from llama_cpp import Llama
            self.llm = Llama.from_pretrained(
                repo_id=self.config.model_name,
                filename="mistral-small-3.1-24b-instruct-2503-hf-q6_k.gguf",
                verbose=False
            )
            self.is_available = True
            return True
        except Exception as e:
            logger.error(f"Failed to load Mistral model: {e}")
            self.is_available = False
            return False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.is_available or not self.llm:
            raise RuntimeError("Local Mistral not available")
        
        start_time = time.time()
        
        try:
            response = self.llm(
                prompt,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                echo=False
            )
            
            response_time = time.time() - start_time
            
            # Extract content
            if isinstance(response, dict) and 'choices' in response:
                content = response['choices'][0]['text']
            else:
                content = str(response)
            
            return LLMResponse(
                content=content.strip(),
                provider=LLMProvider.LOCAL_MISTRAL,
                model_name=self.config.model_name,
                response_time=response_time,
                metadata={'raw_response': response}
            )
            
        except Exception as e:
            logger.error(f"Local Mistral generation failed: {e}")
            raise
    
    def is_running(self) -> bool:
        return self.is_available and self.llm is not None
    
    def start(self) -> bool:
        return self._test_connection()
    
    def stop(self) -> bool:
        self.llm = None
        self.is_available = False
        return True

class CLILLMInterface(BaseLLMInterface):
    """Interface for CLI-based LLMs (Gemini, Claude, etc.)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.process = None
        self.command = self._build_command()
    
    def _build_command(self) -> str:
        """Build the CLI command based on provider"""
        if self.config.command_template:
            return self.config.command_template
        
        commands = {
            LLMProvider.GEMINI_CLI: "gemini",
            LLMProvider.CLAUDE_CLI: "claude",
            LLMProvider.GPT_CLI: "gpt",
            LLMProvider.OLLAMA: f"ollama run {self.config.model_name}"
        }
        
        return commands.get(self.config.provider, "")
    
    def _test_connection(self) -> bool:
        try:
            result = subprocess.run(
                [self.command.split()[0], "--help"],
                capture_output=True,
                text=True,
                timeout=5
            )
            self.is_available = result.returncode == 0
            return self.is_available
        except Exception as e:
            logger.error(f"CLI LLM test failed: {e}")
            self.is_available = False
            return False
    
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if not self.is_available:
            raise RuntimeError(f"{self.config.provider.value} not available")
        
        start_time = time.time()
        
        try:
            # For CLI tools, we need to handle interactive input
            if self.config.provider == LLMProvider.GEMINI_CLI:
                return self._generate_gemini(prompt, **kwargs)
            elif self.config.provider == LLMProvider.OLLAMA:
                return self._generate_ollama(prompt, **kwargs)
            else:
                return self._generate_generic_cli(prompt, **kwargs)
                
        except Exception as e:
            logger.error(f"CLI LLM generation failed: {e}")
            raise
    
    def _generate_gemini(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Gemini CLI"""
        try:
            result = subprocess.run(
                ["gemini", prompt],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - time.time()
            
            if result.returncode == 0:
                content = result.stdout.strip()
            else:
                content = f"Error: {result.stderr.strip()}"
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.GEMINI_CLI,
                model_name="gemini-pro",
                response_time=response_time,
                metadata={'return_code': result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Gemini CLI timeout")
    
    def _generate_ollama(self, prompt: str, **kwargs) -> LLMResponse:
        """Generate response using Ollama"""
        try:
            result = subprocess.run(
                ["ollama", "run", self.config.model_name, prompt],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - time.time()
            
            if result.returncode == 0:
                content = result.stdout.strip()
            else:
                content = f"Error: {result.stderr.strip()}"
            
            return LLMResponse(
                content=content,
                provider=LLMProvider.OLLAMA,
                model_name=self.config.model_name,
                response_time=response_time,
                metadata={'return_code': result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("Ollama timeout")
    
    def _generate_generic_cli(self, prompt: str, **kwargs) -> LLMResponse:
        """Generic CLI generation method"""
        try:
            result = subprocess.run(
                self.command.split() + [prompt],
                capture_output=True,
                text=True,
                timeout=self.config.timeout
            )
            
            response_time = time.time() - time.time()
            
            if result.returncode == 0:
                content = result.stdout.strip()
            else:
                content = f"Error: {result.stderr.strip()}"
            
            return LLMResponse(
                content=content,
                provider=self.config.provider,
                model_name=self.config.model_name,
                response_time=response_time,
                metadata={'return_code': result.returncode}
            )
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{self.config.provider.value} timeout")
    
    def is_running(self) -> bool:
        return self.is_available
    
    def start(self) -> bool:
        return self._test_connection()
    
    def stop(self) -> bool:
        if self.process:
            try:
                self.process.terminate()
                self.process = None
            except:
                pass
        return True

class LLMInterfaceFactory:
    """Factory for creating LLM interfaces"""
    
    @staticmethod
    def create_interface(config: LLMConfig) -> BaseLLMInterface:
        """Create the appropriate LLM interface based on provider"""
        
        if config.provider == LLMProvider.LOCAL_MISTRAL:
            return LocalMistralInterface(config)
        elif config.provider in [LLMProvider.GEMINI_CLI, LLMProvider.CLAUDE_CLI, 
                                LLMProvider.GPT_CLI, LLMProvider.OLLAMA]:
            return CLILLMInterface(config)
        else:
            raise ValueError(f"Unsupported LLM provider: {config.provider}")

# Convenience functions
def create_local_mistral(model_name: str = "WesPro/Mistral-Small-3.1-24B-Instruct-2503-HF-Q6_K-GGUF") -> BaseLLMInterface:
    """Create a local Mistral interface"""
    config = LLMConfig(
        provider=LLMProvider.LOCAL_MISTRAL,
        model_name=model_name
    )
    return LLMInterfaceFactory.create_interface(config)

def create_gemini_cli() -> BaseLLMInterface:
    """Create a Gemini CLI interface"""
    config = LLMConfig(
        provider=LLMProvider.GEMINI_CLI,
        model_name="gemini-pro"
    )
    return LLMInterfaceFactory.create_interface(config)

def create_ollama(model_name: str = "llama2") -> BaseLLMInterface:
    """Create an Ollama interface"""
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_name=model_name
    )
    return LLMInterfaceFactory.create_interface(config) 