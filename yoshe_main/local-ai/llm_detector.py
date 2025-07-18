#!/usr/bin/env python3
"""
LLM Detector - Universal LLM Session Detection Tool
Detects and analyzes LLM sessions with confidence levels and loop detection.
"""

import psutil
import win32gui
import win32process
import win32con
import win32api
import time
import re
import os
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

class LLMType(Enum):
    """Supported LLM types"""
    GEMINI = "gemini"
    CLAUDE = "claude"
    GPT = "gpt"
    MISTRAL = "mistral"
    LLAMA = "llama"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    OLLAMA = "ollama"
    LOCAL_LLM = "local_llm"
    UNKNOWN = "unknown"

@dataclass
class LLMConfidence:
    """Confidence score for LLM detection"""
    llm_type: LLMType
    confidence: float  # 0.0 to 1.0
    indicators: List[str]
    evidence: Dict[str, Any]

@dataclass
class SessionAnalysis:
    """Complete analysis of a detected session"""
    hwnd: int
    title: str
    pid: int
    process_name: str
    cmdline: str
    working_directory: Optional[str]
    llm_confidence: LLMConfidence
    loop_analysis: Optional[Dict[str, Any]]
    session_health: str  # "healthy", "potential_loop", "stuck", "unknown"

class LLMDetector:
    """Universal LLM session detector with confidence scoring"""
    
    def __init__(self):
        # Define LLM-specific indicators with weights
        self.llm_indicators = {
            LLMType.GEMINI: {
                'title_patterns': [
                    (r'gemini', 0.9),
                    (r'google.*ai', 0.7),
                    (r'bard', 0.6),
                ],
                'cmdline_patterns': [
                    (r'gemini', 0.9),
                    (r'google.*ai', 0.7),
                    (r'bard', 0.6),
                ],
                'process_patterns': [
                    (r'gemini', 0.8),
                    (r'google', 0.5),
                ]
            },
            LLMType.CLAUDE: {
                'title_patterns': [
                    (r'claude', 0.9),
                    (r'anthropic', 0.8),
                    (r'claude.*assistant', 0.95),
                ],
                'cmdline_patterns': [
                    (r'claude', 0.9),
                    (r'anthropic', 0.8),
                    (r'claude.*assistant', 0.95),
                ],
                'process_patterns': [
                    (r'claude', 0.8),
                    (r'anthropic', 0.7),
                ]
            },
            LLMType.GPT: {
                'title_patterns': [
                    (r'gpt', 0.9),
                    (r'openai', 0.8),
                    (r'chatgpt', 0.95),
                    (r'gpt.*4', 0.9),
                    (r'gpt.*3', 0.8),
                ],
                'cmdline_patterns': [
                    (r'gpt', 0.9),
                    (r'openai', 0.8),
                    (r'chatgpt', 0.95),
                ],
                'process_patterns': [
                    (r'gpt', 0.8),
                    (r'openai', 0.7),
                ]
            },
            LLMType.MISTRAL: {
                'title_patterns': [
                    (r'mistral', 0.9),
                    (r'mistral.*ai', 0.95),
                ],
                'cmdline_patterns': [
                    (r'mistral', 0.9),
                    (r'mistral.*ai', 0.95),
                ],
                'process_patterns': [
                    (r'mistral', 0.8),
                ]
            },
            LLMType.LLAMA: {
                'title_patterns': [
                    (r'llama', 0.9),
                    (r'llama.*cpp', 0.95),
                    (r'meta.*ai', 0.6),
                ],
                'cmdline_patterns': [
                    (r'llama', 0.9),
                    (r'llama.*cpp', 0.95),
                ],
                'process_patterns': [
                    (r'llama', 0.8),
                ]
            },
            LLMType.OLLAMA: {
                'title_patterns': [
                    (r'ollama', 0.9),
                    (r'ollama.*serve', 0.95),
                ],
                'cmdline_patterns': [
                    (r'ollama', 0.9),
                    (r'ollama.*serve', 0.95),
                ],
                'process_patterns': [
                    (r'ollama', 0.8),
                ]
            },
            LLMType.LOCAL_LLM: {
                'title_patterns': [
                    (r'local.*llm', 0.8),
                    (r'quick.*llm', 0.7),
                    (r'python.*llm', 0.6),
                    (r'local.*ai', 0.7),
                ],
                'cmdline_patterns': [
                    (r'python.*llm', 0.7),
                    (r'local.*llm', 0.8),
                    (r'quick.*llm', 0.7),
                ],
                'process_patterns': [
                    (r'python', 0.5),
                ]
            }
        }
        
        # General AI indicators (lower weight, but can boost confidence)
        self.general_ai_indicators = [
            (r'ai.*assistant', 0.3),
            (r'model:', 0.4),
            (r'>>', 0.2),  # Common CLI prompt
            (r'thinking\.\.\.', 0.5),
            (r'error:', 0.2),
            (r'executing:', 0.3),
            (r'command:', 0.2),
            (r'response:', 0.3),
            (r'generating:', 0.4),
        ]
    
    def get_window_text(self, hwnd) -> str:
        """Get text content from a window"""
        try:
            title = win32gui.GetWindowText(hwnd)
            return title
        except:
            return ""
    
    def is_command_prompt(self, hwnd) -> bool:
        """Check if window is a command prompt"""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            process_name = process.name().lower()
            
            cmd_processes = ['cmd.exe', 'powershell.exe', 'pwsh.exe', 'conhost.exe', 
                           'windowsterminal.exe', 'wt.exe', 'python.exe', 'node.exe']
            
            return any(cmd_proc in process_name for cmd_proc in cmd_processes)
        except:
            return False
    
    def get_process_working_directory(self, pid) -> Optional[str]:
        """Get the working directory of a process"""
        try:
            process = psutil.Process(pid)
            return process.cwd()
        except:
            return None
    
    def get_recent_file_activity(self, directory: str, minutes_back: int = 5) -> List[Dict[str, Any]]:
        """Check for recent file modifications in directory"""
        if not directory or not os.path.exists(directory):
            return []
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)
        recent_files = []
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mtime > cutoff_time:
                            recent_files.append({
                                'path': file_path,
                                'modified': mtime,
                                'size': os.path.getsize(file_path)
                            })
                    except:
                        continue
        except:
            pass
        
        return sorted(recent_files, key=lambda x: x['modified'], reverse=True)
    
    def calculate_llm_confidence(self, title: str, cmdline: str, process_name: str) -> LLMConfidence:
        """Calculate confidence scores for different LLM types"""
        best_match = LLMType.UNKNOWN
        best_confidence = 0.0
        best_indicators = []
        best_evidence = {}
        
        # Check each LLM type
        for llm_type, patterns in self.llm_indicators.items():
            confidence = 0.0
            indicators = []
            evidence = {
                'title_matches': [],
                'cmdline_matches': [],
                'process_matches': []
            }
            
            # Check title patterns
            for pattern, weight in patterns['title_patterns']:
                if re.search(pattern, title.lower()):
                    confidence += weight
                    indicators.append(f"Title: {pattern}")
                    evidence['title_matches'].append((pattern, weight))
            
            # Check command line patterns
            for pattern, weight in patterns['cmdline_patterns']:
                if re.search(pattern, cmdline.lower()):
                    confidence += weight
                    indicators.append(f"Command: {pattern}")
                    evidence['cmdline_matches'].append((pattern, weight))
            
            # Check process patterns
            for pattern, weight in patterns['process_patterns']:
                if re.search(pattern, process_name.lower()):
                    confidence += weight
                    indicators.append(f"Process: {pattern}")
                    evidence['process_matches'].append((pattern, weight))
            
            # Apply general AI indicators (boost confidence)
            for pattern, weight in self.general_ai_indicators:
                if re.search(pattern, title.lower()) or re.search(pattern, cmdline.lower()):
                    confidence += weight * 0.5  # Reduced weight for general indicators
                    indicators.append(f"General AI: {pattern}")
            
            # Cap confidence at 1.0
            confidence = min(confidence, 1.0)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = llm_type
                best_indicators = indicators
                best_evidence = evidence
        
        return LLMConfidence(
            llm_type=best_match,
            confidence=best_confidence,
            indicators=best_indicators,
            evidence=best_evidence
        )
    
    def detect_potential_loop(self, session_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect if a session might be stuck in a loop"""
        working_dir = session_info.get('working_directory')
        if not working_dir:
            return None
        
        recent_files = self.get_recent_file_activity(working_dir, minutes_back=5)
        
        loop_indicators = {
            'no_recent_activity': len(recent_files) == 0,
            'working_directory': working_dir,
            'recent_files_count': len(recent_files),
            'recent_files': recent_files[:5],
            'last_activity': recent_files[0]['modified'] if recent_files else None
        }
        
        return loop_indicators
    
    def analyze_session_health(self, loop_analysis: Optional[Dict[str, Any]], 
                              confidence: float) -> str:
        """Determine session health status"""
        if confidence < 0.3:
            return "unknown"
        
        if not loop_analysis:
            return "unknown"
        
        if loop_analysis.get('no_recent_activity', False):
            return "potential_loop"
        
        recent_files = loop_analysis.get('recent_files_count', 0)
        if recent_files == 0:
            return "stuck"
        
        return "healthy"
    
    def analyze_window(self, hwnd) -> Optional[SessionAnalysis]:
        """Analyze a window for LLM activity"""
        try:
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            process = psutil.Process(pid)
            cmdline = ' '.join(process.cmdline()).lower()
            working_directory = self.get_process_working_directory(pid)
            
            # Calculate LLM confidence
            llm_confidence = self.calculate_llm_confidence(title, cmdline, process.name())
            
            # Only proceed if we have some confidence
            if llm_confidence.confidence < 0.2:
                return None
            
            # Analyze for potential loops
            session_info = {'working_directory': working_directory}
            loop_analysis = self.detect_potential_loop(session_info)
            
            # Determine session health
            session_health = self.analyze_session_health(loop_analysis, llm_confidence.confidence)
            
            return SessionAnalysis(
                hwnd=hwnd,
                title=title,
                pid=pid,
                process_name=process.name(),
                cmdline=cmdline,
                working_directory=working_directory,
                llm_confidence=llm_confidence,
                loop_analysis=loop_analysis,
                session_health=session_health
            )
            
        except Exception as e:
            return None
    
    def find_llm_sessions(self) -> List[SessionAnalysis]:
        """Find all potential LLM sessions"""
        llm_sessions = []
        
        def enum_windows_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                if self.is_command_prompt(hwnd):
                    analysis = self.analyze_window(hwnd)
                    if analysis:
                        results.append(analysis)
            return True
        
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        return windows
    
    def monitor_sessions(self, check_interval: int = 5):
        """Continuously monitor for LLM sessions"""
        print("Starting LLM session monitor...")
        print(f"Checking every {check_interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                sessions = self.find_llm_sessions()
                
                if sessions:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(sessions)} LLM session(s):")
                    for i, session in enumerate(sessions, 1):
                        self._print_session_info(session, i)
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No LLM sessions detected")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")
    
    def _print_session_info(self, session: SessionAnalysis, session_num: int):
        """Print formatted session information"""
        conf = session.llm_confidence
        
        print(f"\n  Session {session_num}:")
        print(f"    Title: {session.title}")
        print(f"    Process: {session.process_name} (PID: {session.pid})")
        print(f"    Working Directory: {session.working_directory}")
        print(f"    LLM Type: {conf.llm_type.value.upper()} (Confidence: {conf.confidence:.2f})")
        print(f"    Session Health: {session.session_health.upper()}")
        print(f"    Indicators: {', '.join(conf.indicators)}")
        
        # Show loop analysis if available
        if session.loop_analysis:
            loop_info = session.loop_analysis
            print(f"    Loop Analysis:")
            print(f"      No recent activity: {loop_info['no_recent_activity']}")
            print(f"      Recent files: {loop_info['recent_files_count']}")
            if loop_info['last_activity']:
                print(f"      Last activity: {loop_info['last_activity'].strftime('%H:%M:%S')}")
            else:
                print(f"      Last activity: None in last 5 minutes")
            
            # Flag potential issues
            if session.session_health == "potential_loop":
                print(f"      ⚠️  POTENTIAL LOOP: No file activity in 5+ minutes")
            elif session.session_health == "stuck":
                print(f"      🚨 SESSION STUCK: No recent file activity")
        
        print(f"    Command line: {session.cmdline[:100]}{'...' if len(session.cmdline) > 100 else ''}")

def main():
    detector = LLMDetector()
    
    # One-time check
    print("Scanning for LLM sessions...")
    sessions = detector.find_llm_sessions()
    
    if sessions:
        print(f"Found {len(sessions)} LLM session(s):")
        for i, session in enumerate(sessions, 1):
            detector._print_session_info(session, i)
            
            # Show recent files if any
            if session.loop_analysis and session.loop_analysis['recent_files']:
                print(f"    Recent files:")
                for file_info in session.loop_analysis['recent_files']:
                    rel_path = os.path.relpath(file_info['path'], session.working_directory)
                    print(f"      {rel_path} ({file_info['modified'].strftime('%H:%M:%S')})")
    else:
        print("No LLM sessions detected")
    
    # Ask if user wants continuous monitoring
    response = input("\nWould you like to start continuous monitoring? (y/n): ")
    if response.lower().startswith('y'):
        detector.monitor_sessions()

if __name__ == "__main__":
    main() 