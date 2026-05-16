import re
import sys
from pathlib import Path

FOR_RE = re.compile(
    r"for\s*\(\s*(?:int|long|size_t|unsigned)?\s*(\w+)\s*=\s*[^;]+;\s*"
    r"\1\s*(?:<|<=)\s*[^;]+;\s*(?:\+\+\1|\1\+\+|\1\s*\+=\s*1)\s*\)\s*\{",
    re.MULTILINE,
)

TYPE_RE = r"(?:int|long|float|double|size_t|unsigned|char)"


def line_number(src, index):
    return src.count("\n", 0, index) + 1


def find_matching_brace(src, open_brace_index):
    depth = 0

    for i in range(open_brace_index, len(src)):
        if src[i] == "{":
            depth += 1
        elif src[i] == "}":
            depth -= 1
            if depth == 0:
                return i

    raise ValueError("Matching brace not found")


def declared_locals(body):
    return set(re.findall(rf"\b{TYPE_RE}\s+(\w+)\b", body))


def is_safe_loop(body, loop_var):
    forbidden = [
        "printf",
        "scanf",
        "malloc",
        "free",
        "fopen",
        "fclose",
        "fprintf",
        "fscanf",
    ]

    for token in forbidden:
        if token in body:
            return False, f"contains side effect: {token}", {}

    if re.search(rf"\[\s*{loop_var}\s*(\+|-)\s*\d+", body):
        return False, "possible loop-carried dependency", {}

    locals_inside = declared_locals(body)
    body_no_headers = re.sub(r"for\s*\([^)]*\)", "", body)

    # Detect reduction patterns: var += expr, var -= expr, etc.
    reductions = {}
    reduction_pattern = r"\b(\w+)\s*(\+=|-=|\*=|/=|%=)\s*"
    
    for match in re.finditer(reduction_pattern, body_no_headers):
        name = match.group(1)
        op_str = match.group(2)
        
        # Only treat as reduction if it's a shared scalar (not loop var, not local)
        if name != loop_var and name not in locals_inside:
            # Map += to +, -= to -, etc.
            op = op_str[0]  # Extract first character (+, -, *, /, %)
            reductions[name] = op

    # Check for loop-carried dependencies (var = var op expr pattern)
    self_assign_pattern = r"\b(\w+)\s*=\s*\1\s*[+\-*/]"
    for match in re.finditer(self_assign_pattern, body_no_headers):
        name = match.group(1)
        if name != loop_var and name not in locals_inside:
            # This is also a reduction
            reductions[name] = "+"

    # If we found reductions, it's still safe
    if reductions:
        return True, f"contains reductions: {', '.join(reductions.keys())}", reductions

    # Check for other shared scalar updates (those are not safe)
    update_patterns = [
        r"\b(\w+)\s*(?:\+\+|--)",
        r"\b(\w+)\s*=\s*[^;{}]*",  # General assignment
    ]

    for pattern in update_patterns:
        for match in re.finditer(pattern, body_no_headers):
            name = match.group(1)
            if name != loop_var and name not in locals_inside:
                # Check if it's a simple increment/decrement
                if "++" in body or "--" in body:
                    if re.search(rf"\b{name}\s*(?:\+\+|--)", body_no_headers):
                        continue  # Allow ++ and -- on shared scalars (they're reductions)
                return False, f"updates possible shared scalar: {name}", {}

    return True, "safe by conservative analysis", {}


def discover_loops(src):
    loops = []

    for match in FOR_RE.finditer(src):
        loop_var = match.group(1)
        open_brace = src.find("{", match.start(), match.end())
        close_brace = find_matching_brace(src, open_brace)

        body = src[open_brace + 1:close_brace]
        safe, reason, reductions = is_safe_loop(body, loop_var)

        loops.append({
            "start": match.start(),
            "end": close_brace + 1,
            "line": line_number(src, match.start()),
            "var": loop_var,
            "safe": safe,
            "reason": reason,
            "reductions": reductions,
        })

    return loops


def choose_outermost_safe_loops(loops):
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
    if "#include <omp.h>" in src:
        return src

    includes = list(re.finditer(r"^#include\s+.*$", src, re.MULTILINE))

    if includes:
        insert_at = includes[-1].end()
        return src[:insert_at] + "\n#include <omp.h>" + src[insert_at:]

    return "#include <omp.h>\n" + src


def transform(src):
    loops = discover_loops(src)
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


def main():
    if len(sys.argv) != 3:
        print("Usage: python tool/auto_parallelizer.py input.c output.c")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    src = input_file.read_text()
    transformed, loops, selected = transform(src)

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
        print(
            f"Line {loop['line']}: for({loop['var']}) -> "
            f"{status} ({loop['reason']}){reduction_info}"
        )

    print()
    print(f"Generated file: {output_file}")


if __name__ == "__main__":
    main()