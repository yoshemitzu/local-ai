import psutil
import win32gui
import win32process
import win32con
import win32api
import time
import re

class GeminiCLIDetector:
    def __init__(self):
        self.gemini_indicators = [
            r'gemini',
            r'claude',
            r'anthropic',
            r'ai assistant',
            r'model:',
            r'>>',  # Common CLI prompt
            r'thinking\.\.\.',
            r'error:',
            r'executing:',
            r'command:',
        ]
        
    def get_window_text(self, hwnd):
        """Get text content from a window"""
        try:
            # Get window title
            title = win32gui.GetWindowText(hwnd)
            
            # Try to get window content (this is limited in Windows)
            # We'll mainly rely on title and process info
            return title
        except:
            return ""
    
    def is_command_prompt(self, hwnd):
        """Check if window is a command prompt"""
        try:
            # Get the process ID
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            # Get process info
            process = psutil.Process(pid)
            process_name = process.name().lower()
            
            # Check for common command prompt processes
            cmd_processes = ['cmd.exe', 'powershell.exe', 'pwsh.exe', 'conhost.exe', 'windowsterminal.exe', 'wt.exe']
            
            return any(cmd_proc in process_name for cmd_proc in cmd_processes)
        except:
            return False
    
    def analyze_window_for_gemini(self, hwnd):
        """Analyze a window to see if it might be running Gemini CLI"""
        try:
            # Get window info
            title = win32gui.GetWindowText(hwnd)
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            
            # Get process info
            process = psutil.Process(pid)
            cmdline = ' '.join(process.cmdline()).lower()
            
            # Check for Gemini indicators
            indicators_found = []
            
            # Check title
            for indicator in self.gemini_indicators:
                if re.search(indicator, title.lower()):
                    indicators_found.append(f"Title: {indicator}")
            
            # Check command line
            for indicator in self.gemini_indicators:
                if re.search(indicator, cmdline):
                    indicators_found.append(f"Command: {indicator}")
            
            return {
                'hwnd': hwnd,
                'title': title,
                'pid': pid,
                'process_name': process.name(),
                'cmdline': cmdline,
                'indicators_found': indicators_found,
                'likely_gemini': len(indicators_found) > 0
            }
            
        except Exception as e:
            return None
    
    def find_gemini_sessions(self):
        """Find all potential Gemini CLI sessions"""
        gemini_sessions = []
        
        def enum_windows_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd):
                if self.is_command_prompt(hwnd):
                    analysis = self.analyze_window_for_gemini(hwnd)
                    if analysis and analysis['likely_gemini']:
                        results.append(analysis)
            return True
        
        # Enumerate all windows
        windows = []
        win32gui.EnumWindows(enum_windows_callback, windows)
        
        return windows
    
    def monitor_sessions(self, check_interval=5):
        """Continuously monitor for Gemini CLI sessions"""
        print("Starting Gemini CLI session monitor...")
        print(f"Checking every {check_interval} seconds")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                sessions = self.find_gemini_sessions()
                
                if sessions:
                    print(f"\n[{time.strftime('%H:%M:%S')}] Found {len(sessions)} potential Gemini CLI session(s):")
                    for i, session in enumerate(sessions, 1):
                        print(f"\n  Session {i}:")
                        print(f"    Title: {session['title']}")
                        print(f"    Process: {session['process_name']} (PID: {session['pid']})")
                        print(f"    Indicators: {', '.join(session['indicators_found'])}")
                        print(f"    Command line: {session['cmdline'][:100]}{'...' if len(session['cmdline']) > 100 else ''}")
                else:
                    print(f"[{time.strftime('%H:%M:%S')}] No Gemini CLI sessions detected")
                
                time.sleep(check_interval)
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped.")

def main():
    detector = GeminiCLIDetector()
    
    # One-time check
    print("Scanning for Gemini CLI sessions...")
    sessions = detector.find_gemini_sessions()
    
    if sessions:
        print(f"Found {len(sessions)} potential Gemini CLI session(s):")
        for i, session in enumerate(sessions, 1):
            print(f"\nSession {i}:")
            print(f"  Title: {session['title']}")
            print(f"  Process: {session['process_name']} (PID: {session['pid']})")
            print(f"  Indicators: {', '.join(session['indicators_found'])}")
            print(f"  Command line: {session['cmdline'][:100]}{'...' if len(session['cmdline']) > 100 else ''}")
    else:
        print("No Gemini CLI sessions detected")
    
    # Ask if user wants continuous monitoring
    response = input("\nWould you like to start continuous monitoring? (y/n): ")
    if response.lower().startswith('y'):
        detector.monitor_sessions()

if __name__ == "__main__":
    main()
