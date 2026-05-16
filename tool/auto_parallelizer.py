import sys
import tempfile
from pathlib import Path

from profiler import profile_hot_functions
from transformer import transform


def main():
    if len(sys.argv) < 3:
        print("Usage: python auto_parallelizer.py input.c output.c [--profile [args]]")
        print("  Optional: --profile to use gprof profiling (specify program args after)")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])
    
    # Check for profiling flag
    use_profiling = "--profile" in sys.argv
    profile_args = None
    
    if use_profiling:
        profile_idx = sys.argv.index("--profile")
        if profile_idx + 1 < len(sys.argv):
            profile_args = sys.argv[profile_idx + 1]

    src = input_file.read_text()
    
    # Profile if requested
    hot_functions = None
    if use_profiling:
        print("\n=== Profiling Phase ===")
        with tempfile.TemporaryDirectory() as tmpdir:
            binary_path = Path(tmpdir) / "profiled_auto_par"
            hot_functions = profile_hot_functions(input_file, binary_path, profile_args)
        print()
    
    # Transform source code with optional profiling
    transformed, loops, selected = transform(src, hot_functions)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(transformed)

    selected_lines = {loop["line"] for loop in selected}

    print("Automatic Parallelization Report")
    print("--------------------------------")

    for loop in loops:
        status = "PARALLELIZED" if loop["line"] in selected_lines else "SKIPPED"
        reduction_info = ""
        if loop.get("reductions"):
            reduction_info = f" [reductions: {', '.join(loop['reductions'].keys())}]"
        func_info = f" in {loop['func']}" if loop['func'] else ""
        print(
            f"Line {loop['line']}: for({loop['var']}) -> "
            f"{status} ({loop['reason']}){reduction_info}{func_info}"
        )

    print()
    print(f"Generated file: {output_file}")


if __name__ == "__main__":
    main()