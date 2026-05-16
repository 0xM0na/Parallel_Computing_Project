"""Gprof-based profiling to identify hot functions."""
import subprocess
import os
from pathlib import Path


def profile_hot_functions(src_path, binary_path, run_args=None, threshold=5.0):
    """
    Profile the code using gprof and return set of hot function names.
    
    Args:
        src_path: Path to source file
        binary_path: Path where to write compiled binary (without .exe extension)
        run_args: Arguments to pass to the program (e.g., "1024")
        threshold: % time threshold to consider "hot" (default 5%)
    
    Returns:
        Set of function names consuming > threshold% of time
    """
    hot_functions = set()
    
    # Ensure proper binary path with .exe on Windows
    binary_path = Path(binary_path)
    if os.name == 'nt' and not str(binary_path).endswith('.exe'):
        binary_path = Path(str(binary_path) + '.exe')
    
    try:
        # Step 1: Compile with profiling flags
        compile_cmd = [
            "gcc",
            "-pg",
            "-O2",
            str(src_path),
            "-o",
            str(binary_path),
            "-lm"
        ]
        
        print("[Profile] Compiling with profiling flags...")
        result = subprocess.run(compile_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[Profile] Compilation failed: {result.stderr}")
            return hot_functions
        
        # Step 2: Run the binary to generate gmon.out
        print(f"[Profile] Running binary...")
        run_cmd = [str(binary_path)]
        if run_args:
            run_cmd.append(str(run_args))
        
        # Run binary from its directory so gmon.out is created there
        binary_dir = str(binary_path.parent)
        result = subprocess.run(run_cmd, capture_output=True, text=True, timeout=60, cwd=binary_dir)
        if result.returncode != 0:
            print(f"[Profile] Execution failed: {result.stderr}")
            return hot_functions
        
        # Step 3: Run gprof and parse output
        print(f"[Profile] Running gprof analysis...")
        gprof_cmd = ["gprof", str(binary_path), "gmon.out"]
        result = subprocess.run(gprof_cmd, capture_output=True, text=True, cwd=binary_dir)
        
        if result.returncode != 0:
            print(f"[Profile] Gprof failed: {result.stderr}")
            return hot_functions
        
        # Parse gprof output to extract hot functions
        gprof_output = result.stdout
        lines = gprof_output.split('\n')
        
        in_table = False
        for line in lines:
            # Look for the flat profile table
            if "% time" in line and "cumulative" in line:
                in_table = True
                continue
            
            if in_table:
                # Stop at the next section
                if line.strip() == "" or "call graph" in line.lower():
                    break
                
                # Parse lines: " 94.5       2.31     2.31        1  2310.00  2310.00  main"
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        percent = float(parts[0])
                        if percent >= threshold:
                            # Function name is typically the last part
                            func_name = parts[-1]
                            if func_name and not func_name.startswith('['):
                                hot_functions.add(func_name)
                    except ValueError:
                        pass
        
        print(f"[Profile] Found {len(hot_functions)} hot functions (>{threshold}% time): {hot_functions}")
        
    except FileNotFoundError:
        print("[Profile] Error: gcc or gprof not found. Skipping profiling.")
    except subprocess.TimeoutExpired:
        print("[Profile] Error: Program execution timed out.")
    except Exception as e:
        print(f"[Profile] Error during profiling: {e}")
    
    return hot_functions
