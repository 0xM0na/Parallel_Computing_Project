"""Code transformation: pragma insertion and OpenMP integration."""
import re
from analyzer import discover_loops


def choose_outermost_safe_loops(loops):
    """
    Select only the outermost safe loops.
    
    Skip nested loops to avoid redundant parallelization.
    """
    selected = []

    for loop in sorted(loops, key=lambda x: x["start"]):
        inside_selected = any(
            chosen["start"] < loop["start"] < chosen["end"]
            for chosen in selected
        )

        if loop["safe"] and not inside_selected:
            selected.append(loop)

    return selected


def add_omp_include(src):
    """Add #include <omp.h> if not already present."""
    if "#include <omp.h>" in src:
        return src

    includes = list(re.finditer(r"^#include\s+.*$", src, re.MULTILINE))

    if includes:
        insert_at = includes[-1].end()
        return src[:insert_at] + "\n#include <omp.h>" + src[insert_at:]

    return "#include <omp.h>\n" + src


def transform(src, hot_functions=None):
    """
    Transform source code by adding OpenMP pragmas to safe loops.
    
    Args:
        src: Source code string
        hot_functions: Set of function names to parallelize (None = all functions)
    
    Returns:
        (transformed_src, loops, selected_loops)
    """
    loops = discover_loops(src, hot_functions)
    selected = choose_outermost_safe_loops(loops)

    transformed = src

    for loop in sorted(selected, key=lambda x: x["start"], reverse=True):
        pragma = "#pragma omp parallel for schedule(static)"
        
        # Add reduction clause if reductions were detected
        if loop.get("reductions"):
            reduction_clauses = ", ".join(
                f"reduction({op}:{var})"
                for var, op in loop["reductions"].items()
            )
            pragma += f" {reduction_clauses}"
        
        transformed = (
            transformed[:loop["start"]]
            + pragma + "\n"
            + transformed[loop["start"]:]
        )

    if selected:
        transformed = add_omp_include(transformed)

    return transformed, loops, selected
