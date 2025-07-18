#!/usr/bin/env python3
"""
AI Swarm Manager - Central Orchestration Hub
Manages detection, spawning, and interaction with multiple LLM instances.
"""

import json
import time
import asyncio
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

from llm_detector import LLMDetector, SessionAnalysis, LLMType
from llm_interface import LLMInterfaceFactory, LLMConfig, LLMProvider, BaseLLMInterface, LLMResponse
from quick_llm import QuickLLM

logger = logging.getLogger(__name__)

@dataclass
class SwarmNode:
    """Represents a single LLM in the swarm"""
    id: str
    llm_type: LLMType
    provider: LLMProvider
    model_name: str
    interface: Optional[BaseLLMInterface]
    session_analysis: Optional[SessionAnalysis]
    status: str  # "detected", "spawned", "active", "inactive", "error"
    confidence: float
    last_activity: datetime
    metadata: Dict[str, Any]

@dataclass
class SwarmTask:
    """Represents a task to be executed by the swarm"""
    id: str
    prompt: str
    target_llms: Optional[List[str]] = None  # None = all available
    priority: int = 1  # 1-10, higher = more important
    timeout: int = 30
    expected_format: str = "text"  # text, json, yaml, list, dict
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class SwarmResult:
    """Result from a swarm task execution"""
    task_id: str
    node_id: str
    response: Optional[LLMResponse]
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0

class AISwarmManager:
    """Central manager for AI swarm operations"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "swarm_config.json"
        self.nodes: Dict[str, SwarmNode] = {}
        self.tasks: List[SwarmTask] = []
        self.results: List[SwarmResult] = []
        
        # Core components
        self.detector = LLMDetector()
        self.quick_llm = QuickLLM()
        
        # Configuration
        self.config = self._load_config()
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Auto-spawn settings
        self.auto_spawn = self.config.get('auto_spawn', True)
        self.preferred_models = self.config.get('preferred_models', [])
        
        logger.info("AI Swarm Manager initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file"""
        default_config = {
            'auto_spawn': True,
            'preferred_models': [
                {'provider': 'local_mistral', 'model_name': 'WesPro/Mistral-Small-3.1-24B-Instruct-2503-HF-Q6_K-GGUF'},
                {'provider': 'gemini_cli', 'model_name': 'gemini-pro'},
                {'provider': 'ollama', 'model_name': 'llama2'}
            ],
            'monitoring_interval': 10,
            'max_nodes': 10,
            'task_timeout': 30
        }
        
        try:
            if Path(self.config_path).exists():
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Merge with defaults
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            else:
                # Create default config
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=2)
                return default_config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return default_config
    
    def scan_system(self) -> List[SessionAnalysis]:
        """Scan system for existing LLM sessions"""
        logger.info("Scanning system for LLM sessions...")
        sessions = self.detector.find_llm_sessions()
        
        for session in sessions:
            self._register_detected_session(session)
        
        logger.info(f"Found {len(sessions)} LLM sessions")
        return sessions
    
    def _register_detected_session(self, session: SessionAnalysis):
        """Register a detected session as a swarm node"""
        node_id = f"detected_{session.pid}_{session.llm_confidence.llm_type.value}"
        
        # Map LLM type to provider
        provider_map = {
            LLMType.GEMINI: LLMProvider.GEMINI_CLI,
            LLMType.CLAUDE: LLMProvider.CLAUDE_CLI,
            LLMType.GPT: LLMProvider.GPT_CLI,
            LLMType.MISTRAL: LLMProvider.LOCAL_MISTRAL,
            LLMType.LLAMA: LLMProvider.OLLAMA,
            LLMType.OLLAMA: LLMProvider.OLLAMA,
            LLMType.LOCAL_LLM: LLMProvider.LOCAL_MISTRAL
        }
        
        provider = provider_map.get(session.llm_confidence.llm_type, LLMProvider.CUSTOM)
        
        node = SwarmNode(
            id=node_id,
            llm_type=session.llm_confidence.llm_type,
            provider=provider,
            model_name=f"{session.llm_confidence.llm_type.value}_detected",
            interface=None,  # Will be created on demand
            session_analysis=session,
            status="detected",
            confidence=session.llm_confidence.confidence,
            last_activity=datetime.now(),
            metadata={
                'pid': session.pid,
                'working_directory': session.working_directory,
                'session_health': session.session_health
            }
        )
        
        self.nodes[node_id] = node
        logger.info(f"Registered detected node: {node_id} ({session.llm_confidence.llm_type.value})")
    
    def spawn_llm(self, provider: LLMProvider, model_name: str, **kwargs) -> Optional[str]:
        """Spawn a new LLM instance"""
        try:
            config = LLMConfig(
                provider=provider,
                model_name=model_name,
                **kwargs
            )
            
            interface = LLMInterfaceFactory.create_interface(config)
            
            if interface.is_available:
                node_id = f"spawned_{provider.value}_{model_name}_{int(time.time())}"
                
                node = SwarmNode(
                    id=node_id,
                    llm_type=self._provider_to_llm_type(provider),
                    provider=provider,
                    model_name=model_name,
                    interface=interface,
                    session_analysis=None,
                    status="spawned",
                    confidence=1.0,  # We spawned it, so we're confident
                    last_activity=datetime.now(),
                    metadata={'spawned_at': datetime.now().isoformat()}
                )
                
                self.nodes[node_id] = node
                logger.info(f"Spawned new LLM node: {node_id}")
                return node_id
            else:
                logger.error(f"Failed to spawn LLM: {provider.value}/{model_name}")
                return None
                
        except Exception as e:
            logger.error(f"Error spawning LLM: {e}")
            return None
    
    def _provider_to_llm_type(self, provider: LLMProvider) -> LLMType:
        """Convert provider to LLM type"""
        mapping = {
            LLMProvider.GEMINI_CLI: LLMType.GEMINI,
            LLMProvider.CLAUDE_CLI: LLMType.CLAUDE,
            LLMProvider.GPT_CLI: LLMType.GPT,
            LLMProvider.LOCAL_MISTRAL: LLMType.MISTRAL,
            LLMProvider.OLLAMA: LLMType.OLLAMA
        }
        return mapping.get(provider, LLMType.UNKNOWN)
    
    def auto_spawn_preferred(self) -> List[str]:
        """Auto-spawn preferred models if not already available"""
        if not self.auto_spawn:
            return []
        
        spawned_nodes = []
        
        for model_config in self.preferred_models:
            provider = LLMProvider(model_config['provider'])
            model_name = model_config['model_name']
            
            # Check if we already have this type
            existing = [node for node in self.nodes.values() 
                       if node.provider == provider and node.status in ['active', 'spawned']]
            
            if not existing:
                node_id = self.spawn_llm(provider, model_name)
                if node_id:
                    spawned_nodes.append(node_id)
        
        return spawned_nodes
    
    def execute_task(self, task: SwarmTask) -> List[SwarmResult]:
        """Execute a task across the swarm"""
        logger.info(f"Executing task {task.id}: {task.prompt[:50]}...")
        
        # Determine target nodes
        target_nodes = []
        if task.target_llms:
            target_nodes = [self.nodes[node_id] for node_id in task.target_llms 
                          if node_id in self.nodes]
        else:
            target_nodes = [node for node in self.nodes.values() 
                          if node.status in ['active', 'spawned']]
        
        if not target_nodes:
            logger.warning("No available nodes for task execution")
            return []
        
        results = []
        
        # Execute task on each target node
        for node in target_nodes:
            try:
                result = self._execute_on_node(task, node)
                results.append(result)
            except Exception as e:
                logger.error(f"Task execution failed on node {node.id}: {e}")
                results.append(SwarmResult(
                    task_id=task.id,
                    node_id=node.id,
                    response=None,
                    success=False,
                    error_message=str(e)
                ))
        
        self.results.extend(results)
        return results
    
    def _execute_on_node(self, task: SwarmTask, node: SwarmNode) -> SwarmResult:
        """Execute a task on a specific node"""
        start_time = time.time()
        
        try:
            # Get or create interface
            if not node.interface:
                config = LLMConfig(
                    provider=node.provider,
                    model_name=node.model_name
                )
                node.interface = LLMInterfaceFactory.create_interface(config)
            
            # Execute the task
            response = node.interface.generate(
                task.prompt,
                max_tokens=512,
                temperature=0.7
            )
            
            execution_time = time.time() - start_time
            
            # Update node status
            node.status = "active"
            node.last_activity = datetime.now()
            
            return SwarmResult(
                task_id=task.id,
                node_id=node.id,
                response=response,
                success=True,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            node.status = "error"
            
            return SwarmResult(
                task_id=task.id,
                node_id=node.id,
                response=None,
                success=False,
                error_message=str(e),
                execution_time=execution_time
            )
    
    def start_monitoring(self, interval: int = 10):
        """Start continuous monitoring of the swarm"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("Swarm monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("Swarm monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Scan for new sessions
                new_sessions = self.scan_system()
                
                # Update existing nodes
                self._update_node_statuses()
                
                # Auto-spawn if needed
                if self.auto_spawn:
                    self.auto_spawn_preferred()
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _update_node_statuses(self):
        """Update status of existing nodes"""
        for node in self.nodes.values():
            if node.interface:
                try:
                    if node.interface.is_running():
                        node.status = "active"
                    else:
                        node.status = "inactive"
                except:
                    node.status = "error"
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Get comprehensive swarm status"""
        status = {
            'total_nodes': len(self.nodes),
            'active_nodes': len([n for n in self.nodes.values() if n.status == 'active']),
            'detected_nodes': len([n for n in self.nodes.values() if n.status == 'detected']),
            'spawned_nodes': len([n for n in self.nodes.values() if n.status == 'spawned']),
            'error_nodes': len([n for n in self.nodes.values() if n.status == 'error']),
            'total_tasks': len(self.tasks),
            'total_results': len(self.results),
            'monitoring_active': self.monitoring_active,
            'nodes': {node_id: asdict(node) for node_id, node in self.nodes.items()}
        }
        return status
    
    def print_swarm_status(self):
        """Print formatted swarm status"""
        status = self.get_swarm_status()
        
        print("\n🤖 AI SWARM STATUS")
        print("=" * 50)
        print(f"Total Nodes: {status['total_nodes']}")
        print(f"Active: {status['active_nodes']} | Detected: {status['detected_nodes']} | Spawned: {status['spawned_nodes']} | Errors: {status['error_nodes']}")
        print(f"Tasks: {status['total_tasks']} | Results: {status['total_results']}")
        print(f"Monitoring: {'🟢 Active' if status['monitoring_active'] else '🔴 Inactive'}")
        
        if status['nodes']:
            print("\n📋 NODES:")
            for node_id, node_data in status['nodes'].items():
                status_emoji = {
                    'active': '🟢',
                    'detected': '🔍',
                    'spawned': '🟡',
                    'inactive': '⚪',
                    'error': '🔴'
                }.get(node_data['status'], '❓')
                
                print(f"  {status_emoji} {node_id[:20]:<20} | {node_data['llm_type']:<10} | {node_data['status']:<10} | Conf: {node_data['confidence']:.2f}")

def main():
    """Main entry point for AI Swarm Manager"""
    print("🚀 Starting AI Swarm Manager...")
    
    # Initialize swarm manager
    swarm = AISwarmManager()
    
    # Initial system scan
    print("\n🔍 Scanning system for existing LLM sessions...")
    sessions = swarm.scan_system()
    
    if sessions:
        print(f"Found {len(sessions)} existing LLM sessions")
    else:
        print("No existing LLM sessions found")
    
    # Auto-spawn preferred models
    print("\n🤖 Auto-spawning preferred models...")
    spawned = swarm.auto_spawn_preferred()
    if spawned:
        print(f"Spawned {len(spawned)} new LLM instances")
    else:
        print("No new instances spawned")
    
    # Show initial status
    swarm.print_swarm_status()
    
    # Start monitoring
    print("\n📊 Starting continuous monitoring...")
    swarm.start_monitoring(interval=10)
    
    try:
        # Interactive mode
        print("\n💬 Interactive mode - Type 'help' for commands")
        while True:
            command = input("\nswarm> ").strip().lower()
            
            if command == 'help':
                print("""
Available commands:
  status     - Show swarm status
  scan       - Scan for new sessions
  spawn      - Spawn new LLM
  task       - Execute task
  stop       - Stop monitoring and exit
  help       - Show this help
                """)
            
            elif command == 'status':
                swarm.print_swarm_status()
            
            elif command == 'scan':
                sessions = swarm.scan_system()
                print(f"Scan complete. Found {len(sessions)} sessions.")
            
            elif command == 'spawn':
                print("Available providers: local_mistral, gemini_cli, ollama")
                provider = input("Provider: ").strip()
                model = input("Model name: ").strip()
                
                try:
                    provider_enum = LLMProvider(provider)
                    node_id = swarm.spawn_llm(provider_enum, model)
                    if node_id:
                        print(f"Spawned: {node_id}")
                    else:
                        print("Failed to spawn")
                except ValueError:
                    print("Invalid provider")
            
            elif command == 'task':
                prompt = input("Enter task prompt: ").strip()
                if prompt:
                    task = SwarmTask(
                        id=f"task_{int(time.time())}",
                        prompt=prompt
                    )
                    results = swarm.execute_task(task)
                    print(f"Task executed on {len(results)} nodes")
                    
                    for result in results:
                        if result.success and result.response:
                            print(f"\n✅ {result.node_id}:")
                            print(f"   {result.response.content[:200]}...")
                        else:
                            print(f"\n❌ {result.node_id}: {result.error_message}")
            
            elif command == 'stop':
                break
            
            else:
                print("Unknown command. Type 'help' for available commands.")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down...")
    
    finally:
        swarm.stop_monitoring()
        print("AI Swarm Manager stopped.")

if __name__ == "__main__":
    main() 