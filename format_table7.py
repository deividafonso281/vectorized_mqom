header = r"""\begin{table}[H]\centering\color{black}
\renewcommand{\arraystretch}{0.70}\setlength{\tabcolsep}{1.6pt}
\begin{tabular}{@{}clc rr rr rr@{}}
\toprule
& & & \multicolumn{2}{c}{Key Gen} & \multicolumn{2}{c}{Sign} & \multicolumn{2}{c}{Verify} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Level & Variant & Tree & Ours & Ref. & Ours & Ref. & Ours & Ref. \\
\midrule"""

footer = r"""\end{tabular}
\caption{MQOM v3 benchmarks on AVX2+VAES (millions of cycles; lower is better; bold marks the lower Ours/Reference value).}\label{tab:AVX2v3}
\end{table}"""

from pathlib import Path
import re

root = Path(".")

roman = {}
roman["1"] = "I"
roman["3"] = "III"
roman["5"] = "V"

fields = {}
fields["gf2"] = "$GF(2)$"
fields["gf16"] = "$GF(2^4)$"

print(header)

for level in ["1", "3", "5"]:
    print(r"""\multirow{6}{*}""",end="")
    print(f"{{{roman[level]}}}")
    for variation in ["gf2_shorter","gf16_fast","gf16_short"]:
        field = variation.split('_')[0]
        speed = variation.split('_')[1]
        tree_types = {}
        tree_types["ct"] = f"& \\multirow{{2}}{{*}}{{{fields[field]} {speed}}} & CT & "
        tree_types["ot"] = "& & OT & "
        for tree_type in ["ct", "ot"]:
            with open(f"./Row-wise_v3/Optimized_Implementation/mqom3_cat{level}_{variation}_{tree_type}/bench_avx2_gcc.txt", "r", encoding="utf-8") as optimized_rowf:
                conteudo_row_f = optimized_rowf.read()

            keygen_row = float(re.findall(r"^ - Key Gen: (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            sign_row = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            verify_row = float(re.findall(r"^ - Verify:  (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            memory_row = int(re.findall(r"^\[\+\] Malloc stats keygen\+sign\+verif: Peak Usage is (.*) bytes$", conteudo_row_f, flags=re.MULTILINE)[0])

            with open(f"./Reference_v3/Optimized_Implementation/mqom3_cat{level}_{variation}_{tree_type}/bench_avx2_gcc.txt", "r", encoding="utf-8") as submissionf:
                conteudo_submission = submissionf.read()

            keygen_submission = float(re.findall(r"^ - Key Gen: (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            sign_submission = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            verify_submission = float(re.findall(r"^ - Verify:  (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            memory_submission = int(re.findall(r"^\[\+\] Malloc stats keygen\+sign\+verif: Peak Usage is (.*) bytes$", conteudo_submission, flags=re.MULTILINE)[0])

            print(f"{tree_types[tree_type]}", end='')
            print(f" {keygen_row:.2f} & {keygen_submission:.2f} & {sign_row:.2f} & {sign_submission:.2f} & {verify_row:.2f} & {verify_submission:.2f}\\\\")
        
        if variation != "gf16_short":
            print("\\addlinespace[1pt]")
        
    if level == "5":
        print("\\bottomrule")
    else:
        print("\\midrule")

print(footer)
