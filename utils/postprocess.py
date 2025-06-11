import numpy as np

def sobol_to_latex(cols, ranges, first_order, total_order, filename="sobol_indices_table.tex"):
    with open(filename, "w") as f:
        f.write("\\begin{table}[ht]\n\\centering\n")
        f.write("\\begin{tabular}{l" + "c" * (len(cols)+1) + "}\n")
        f.write("\\hline\n")
        # Header
        f.write("PARAMETER & " + " & ".join([name for name in cols]) + " & SUM \\\\\n")
        f.write("\\hline\n")
        # Ranges
        f.write("RANGE & " + " & ".join(ranges) + " & \\\\\n")
        # First-order
        f.write("FIRST-ORDER & " + " & ".join([f"{v:.4f}" for v in first_order]) + f" & {np.sum(first_order):.4f} \\\\\n")
        # Total-order
        f.write("TOTAL-ORDER & " + " & ".join([f"{v:.4f}" for v in total_order]) + f" & {np.sum(total_order):.4f} \\\\\n")
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{Sobol indices for the output variable.}\n")
        f.write("\\end{table}\n")
