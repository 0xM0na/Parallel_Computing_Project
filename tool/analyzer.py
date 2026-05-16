"""Code analysis: loop detection and safety checking."""
import re

FOR_RE = re.compile(
    r"for\s*\(\s*(?:int|long|size_t|unsigned)?\s*(\w+)\s*=\s*[^;]+;\s*"
    r"\1\s*(?:<|<=)\s*[^;]+;\s*(?:\+\+\1|\1\+\+|\1\s*\+=\s*1)\s*\)\s*\{",
    re.MULTILINE,
)

TYPE_RE = r"(?:int|long|float|double|size_t|unsigned|char)"


def line_number(src, index):
    """Convert source position index to line number."""
    return src.count("\n", 0, index) + 1


def find_matching_brace(src, open_brace_index):
    """Find the closing brace that matches the opening brace at given index."""
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
    """Find all local variables declared in the given code block."""
    return set(re.findall(rf"\b{TYPE_RE}\s+(\w+)\b", body))


def find_enclosing_function(src, position):
    """Find the function name that contains the given position in source code."""
    func_pattern = r"^(?:int|void|double|float|char|long|unsigned|static)*\s+(\w+)\s*\([^)]*\)\s*\{"
    
    # Search backwards line by line
    text_before = src[:position]
    lines_before = text_before.split('\n')
    
    for i in range(len(lines_before) - 1, -1, -1):
        line = lines_before[i]
        match = re.search(func_pattern, line)
        if match:
            return match.group(1)
    
    return None  # Top level or couldn't determine


def is_safe_loop(body, loop_var):
    """
    Analyze a loop body for safety.
    
    Returns:
        (is_safe, reason, reductions_dict)
    """
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


def discover_loops(src, hot_functions=None):
    """
    Discover all loops in source code.
    
    Args:
        src: Source code string
        hot_functions: Set of function names to parallelize (None = all functions)
    
    Returns:
        List of loop dictionaries with metadata
    """
    loops = []

    for match in FOR_RE.finditer(src):
        loop_var = match.group(1)
        open_brace = src.find("{", match.start(), match.end())
        close_brace = find_matching_brace(src, open_brace)

        body = src[open_brace + 1:close_brace]
        safe, reason, reductions = is_safe_loop(body, loop_var)

        # Determine which function this loop belongs to
        func_name = find_enclosing_function(src, match.start())
        
        # If profiling data available, check if function is hot
        skip_cold = False
        if hot_functions is not None and func_name not in hot_functions:
            skip_cold = True
            if func_name:
                reason = f"in cold function: {func_name}"
            else:
                reason = "function not identified"
        
        loops.append({
            "start": match.start(),
            "end": close_brace + 1,
            "line": line_number(src, match.start()),
            "var": loop_var,
            "safe": safe and not skip_cold,
            "reason": reason,
            "reductions": reductions,
            "func": func_name,
        })

    return loops
