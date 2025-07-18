#!/usr/bin/env python3
"""
Local AI - AI Swarm Management System
Main entry point for the complete AI orchestration platform.
"""

import sys
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point with command-line interface"""
    parser = argparse.ArgumentParser(
        description="Local AI - AI Swarm Management System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py swarm                    # Start AI swarm manager
  python main.py detect                   # Scan for LLM sessions
  python main.py quick "Hello world"      # Quick LLM query
  python main.py monitor                  # Start monitoring mode
        """
    )
    
    parser.add_argument(
        'command',
        choices=['swarm', 'detect', 'quick', 'monitor', 'help'],
        help='Command to execute'
    )
    
    parser.add_argument(
        'args',
        nargs='*',
        help='Additional arguments for the command'
    )
    
    parser.add_argument(
        '--config',
        default='swarm_config.json',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.command == 'swarm':
            run_swarm_manager(args.config)
        elif args.command == 'detect':
            run_detector()
        elif args.command == 'quick':
            if not args.args:
                print("Error: Quick command requires a prompt")
                sys.exit(1)
            run_quick_llm(' '.join(args.args))
        elif args.command == 'monitor':
            run_monitor()
        elif args.command == 'help':
            show_help()
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n🛑 Operation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)

def run_swarm_manager(config_path: str):
    """Run the AI swarm manager"""
    print("🚀 Starting AI Swarm Manager...")
    
    try:
        from swarm_manager import AISwarmManager
        
        swarm = AISwarmManager(config_path)
        
        # Initial scan and setup
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
        
        # Interactive mode
        print("\n💬 Interactive mode - Type 'help' for commands")
        interactive_loop(swarm)
        
    except ImportError as e:
        print(f"Error: Missing required module - {e}")
        print("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

def interactive_loop(swarm):
    """Interactive command loop for swarm manager"""
    import time
    from swarm_manager import SwarmTask
    from llm_interface import LLMProvider
    
    while True:
        try:
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
            break
        except Exception as e:
            print(f"Error: {e}")
    
    swarm.stop_monitoring()
    print("AI Swarm Manager stopped.")

def run_detector():
    """Run the LLM detector"""
    print("🔍 Running LLM Detector...")
    
    try:
        from llm_detector import LLMDetector
        
        detector = LLMDetector()
        sessions = detector.find_llm_sessions()
        
        if sessions:
            print(f"\nFound {len(sessions)} LLM session(s):")
            for i, session in enumerate(sessions, 1):
                detector._print_session_info(session, i)
        else:
            print("No LLM sessions detected")
        
        # Ask if user wants continuous monitoring
        response = input("\nWould you like to start continuous monitoring? (y/n): ")
        if response.lower().startswith('y'):
            detector.monitor_sessions()
            
    except ImportError as e:
        print(f"Error: Missing required module - {e}")
        sys.exit(1)

def run_quick_llm(prompt: str):
    """Run a quick LLM query"""
    print(f"⚡ Quick LLM Query: {prompt[:50]}...")
    
    try:
        from quick_llm import quick_llm_call
        
        response = quick_llm_call(prompt, "text")
        print(f"\n🤖 Response:\n{response}")
        
    except ImportError as e:
        print(f"Error: Missing required module - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def run_monitor():
    """Run monitoring mode"""
    print("📊 Starting LLM Monitoring...")
    
    try:
        from llm_detector import LLMDetector
        
        detector = LLMDetector()
        detector.monitor_sessions()
        
    except ImportError as e:
        print(f"Error: Missing required module - {e}")
        sys.exit(1)

def show_help():
    """Show comprehensive help"""
    help_text = """
🤖 Local AI - AI Swarm Management System

This system provides comprehensive AI orchestration capabilities:

COMPONENTS:
  🚀 Swarm Manager    - Central orchestration hub
  🔍 LLM Detector     - Universal LLM session detection
  ⚡ Quick LLM        - Lightweight local LLM utility
  📊 Monitor          - Real-time session monitoring

COMMANDS:
  swarm               - Start AI swarm manager (interactive)
  detect              - Scan for LLM sessions
  quick <prompt>      - Execute quick LLM query
  monitor             - Start monitoring mode
  help                - Show this help

EXAMPLES:
  python main.py swarm                    # Start full swarm management
  python main.py detect                   # Scan for existing LLMs
  python main.py quick "Hello world"      # Quick query
  python main.py monitor                  # Monitor sessions

FEATURES:
  ✅ Multi-LLM detection (Gemini, Claude, GPT, Mistral, etc.)
  ✅ Confidence-based identification
  ✅ Auto-spawning of preferred models
  ✅ Task distribution across swarm
  ✅ Real-time monitoring and health checks
  ✅ Structured output support
  ✅ Local and remote LLM integration

For more information, see the roadmap.md file.
    """
    print(help_text)

if __name__ == "__main__":
    main() 