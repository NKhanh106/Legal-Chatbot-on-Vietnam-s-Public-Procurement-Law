"""
Script để khởi động cả backend và frontend cùng lúc.
Sử dụng subprocess để chạy cả 2 processes song song.
"""
import subprocess
import sys
import os
import signal
import time
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()
BACKEND_DIR = PROJECT_ROOT / "backend" / "api"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

def check_dependencies():
    """Kiểm tra dependencies cần thiết."""
    print("🔍 Checking dependencies...")
    
    # Check Python
    try:
        python_version = sys.version_info
        print(f"   ✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    except Exception as e:
        print(f"   ❌ Python check failed: {e}")
        return False
    
    # Check Node.js
    try:
        result = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"   ✅ Node.js {result.stdout.strip()}")
        else:
            print("   ⚠️  Node.js not found (frontend sẽ không chạy)")
            return False
    except FileNotFoundError:
        print("   ⚠️  Node.js not found (frontend sẽ không chạy)")
        return False
    except Exception as e:
        print(f"   ⚠️  Node.js check failed: {e}")
        return False
    
    # Check npm (npm thường đi kèm với Node.js)
    try:
        # Thử nhiều cách để tìm npm
        npm_commands = ["npm", "npm.cmd"]  # npm.cmd cho Windows
        npm_found = False
        
        for npm_cmd in npm_commands:
            try:
                result = subprocess.run(
                    [npm_cmd, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    shell=True  # Dùng shell=True trên Windows để tìm trong PATH
                )
                if result.returncode == 0:
                    print(f"   ✅ npm {result.stdout.strip()}")
                    npm_found = True
                    break
            except FileNotFoundError:
                continue
            except Exception:
                continue
        
        if not npm_found:
            # Nếu không tìm thấy npm nhưng có Node.js, có thể npm vẫn hoạt động
            # (npm thường đi kèm với Node.js)
            print("   ⚠️  npm command not found in PATH, but will try anyway (npm usually comes with Node.js)")
            # Không return False, vì có thể npm vẫn hoạt động khi chạy từ frontend folder
    except Exception as e:
        print(f"   ⚠️  npm check failed: {e}, but will try anyway")
    
    return True

def start_backend():
    """Khởi động backend server."""
    print("\n🚀 Starting Backend Server...")
    print(f"   📁 Directory: {BACKEND_DIR}")
    print(f"   🌐 URL: http://localhost:5000")
    
    # Change to backend directory
    os.chdir(BACKEND_DIR)
    
    # Start Flask server
    # Set UTF-8 encoding để tránh lỗi Unicode trên Windows PowerShell
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    
    backend_process = subprocess.Popen(
        [sys.executable, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
        encoding='utf-8',
        errors='replace'  # Replace invalid characters thay vì crash
    )
    
    return backend_process

def start_frontend():
    """Khởi động frontend dev server."""
    print("\n🚀 Starting Frontend Dev Server...")
    print(f"   📁 Directory: {FRONTEND_DIR}")
    print(f"   🌐 URL: http://localhost:5173")
    
    # Check if node_modules exists
    if not (FRONTEND_DIR / "node_modules").exists():
        print("   ⚠️  node_modules not found. Installing dependencies...")
        print("   📦 Running: npm install")
        
        # Thử nhiều cách để chạy npm install
        npm_commands = ["npm", "npm.cmd"]
        install_success = False
        
        for npm_cmd in npm_commands:
            try:
                install_process = subprocess.run(
                    [npm_cmd, "install"],
                    cwd=str(FRONTEND_DIR),
                    shell=True if sys.platform == 'win32' else False,
                    timeout=300  # 5 minutes timeout
                )
                if install_process.returncode == 0:
                    install_success = True
                    break
            except FileNotFoundError:
                continue
            except Exception as e:
                print(f"   ⚠️  Failed with {npm_cmd}: {e}")
                continue
        
        if not install_success:
            # Fallback: thử với shell command
            try:
                if sys.platform == 'win32':
                    install_process = subprocess.run(
                        "npm install",
                        cwd=str(FRONTEND_DIR),
                        shell=True,
                        timeout=300
                    )
                else:
                    install_process = subprocess.run(
                        ["npm", "install"],
                        cwd=str(FRONTEND_DIR),
                        timeout=300
                    )
                if install_process.returncode != 0:
                    print("   ❌ npm install failed!")
                    return None
            except Exception as e:
                print(f"   ❌ npm install failed: {e}")
                return None
    
    # Change to frontend directory
    os.chdir(FRONTEND_DIR)
    
    # Start Vite dev server
    # Thử nhiều cách để chạy npm (Windows có thể cần npm.cmd hoặc shell=True)
    npm_commands = ["npm", "npm.cmd"]
    frontend_process = None
    
    for npm_cmd in npm_commands:
        try:
            frontend_process = subprocess.Popen(
                [npm_cmd, "run", "dev"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                shell=True if sys.platform == 'win32' else False
            )
            # Test xem process có chạy được không
            time.sleep(0.5)
            if frontend_process.poll() is None:  # Process vẫn đang chạy
                break
            else:
                frontend_process = None
        except FileNotFoundError:
            continue
        except Exception:
            continue
    
    if not frontend_process:
        # Fallback: thử với shell command trực tiếp
        try:
            if sys.platform == 'win32':
                frontend_process = subprocess.Popen(
                    "npm run dev",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    shell=True,
                    cwd=str(FRONTEND_DIR)
                )
            else:
                frontend_process = subprocess.Popen(
                    ["npm", "run", "dev"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    universal_newlines=True,
                    cwd=str(FRONTEND_DIR)
                )
        except Exception as e:
            print(f"   ❌ Failed to start frontend: {e}")
            return None
    
    return frontend_process

def print_output(process, prefix):
    """In output từ process với prefix."""
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                try:
                    print(f"[{prefix}] {line.rstrip()}")
                except UnicodeEncodeError:
                    # Fallback: loại bỏ emoji nếu không thể encode trên Windows
                    safe_line = line.encode('ascii', 'ignore').decode('ascii')
                    print(f"[{prefix}] {safe_line.rstrip()}")
    except Exception:
        pass

def main():
    """Main function."""
    print("=" * 60)
    print("🚀 LEGAL CHATBOT - DEVELOPMENT SERVER")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed. Please install required dependencies.")
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("\n❌ Failed to start backend!")
        sys.exit(1)
    
    # Wait a bit for backend to start
    time.sleep(2)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("\n❌ Failed to start frontend!")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✅ Both servers are starting...")
    print("=" * 60)
    print("\n📡 Backend:  http://localhost:5000")
    print("🌐 Frontend: http://localhost:5173")
    print("\n💡 Press Ctrl+C to stop both servers")
    print("=" * 60 + "\n")
    
    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down servers...")
        if backend_process:
            backend_process.terminate()
        if frontend_process:
            frontend_process.terminate()
        print("✅ Servers stopped. Goodbye!")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Monitor processes and print output
    try:
        import threading
        
        # Start threads to print output
        backend_thread = threading.Thread(
            target=print_output,
            args=(backend_process, "BACKEND"),
            daemon=True
        )
        frontend_thread = threading.Thread(
            target=print_output,
            args=(frontend_process, "FRONTEND"),
            daemon=True
        )
        
        backend_thread.start()
        frontend_thread.start()
        
        # Wait for processes to finish
        while True:
            backend_code = backend_process.poll()
            frontend_code = frontend_process.poll()
            
            if backend_code is not None:
                print(f"\n⚠️  Backend process exited with code {backend_code}")
                if frontend_process.poll() is None:
                    frontend_process.terminate()
                break
            
            if frontend_code is not None:
                print(f"\n⚠️  Frontend process exited with code {frontend_code}")
                if backend_process.poll() is None:
                    backend_process.terminate()
                break
            
            time.sleep(1)
    
    except KeyboardInterrupt:
        signal_handler(None, None)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    main()

