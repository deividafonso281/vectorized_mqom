header = r"""\begin{table}[H]\centering\color{black}\scriptsize
\renewcommand{\arraystretch}{1.00}\setlength{\tabcolsep}{2.15pt}
\begin{tabular}{@{}cll rr rr rr@{}}
\toprule
& & & \multicolumn{2}{c}{Key Gen} & \multicolumn{2}{c}{Sign} & \multicolumn{2}{c}{Verify} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Platform & Level & Variant & Ours & Ref. & Ours & Ref. & Ours & Ref. \\
\midrule"""

footer = r"""\end{tabular}
\caption{End-to-end MQOM v2.1 timings over $GF(2^4)$, in millions of cycles. Bold marks the lower (better) value in each Ours/Reference pair; ties are left unbolded.}\label{tab:timesGF16combined}
\end{table}"""

from pathlib import Path
import re

root = Path(".")

roman = {}
roman["1"] = "I"
roman["3"] = "III"
roman["5"] = "V"

paths_reference = {}
paths_opt = {}

paths_reference["NEON"] = "./NEON/Multiplication_NOT_Transposed/"
paths_opt["NEON"] = "./NEON/Row-wise/"
paths_reference["AVX2"] = "./AVX/Reference/"
paths_opt["AVX2"] = "./AVX/Row-wise/"
paths_reference["AVX-512+GFNI"] = "./AVX/Reference/"
paths_opt["AVX-512+GFNI"] = "./AVX/Row-wise/"

output_files = {}
output_files["NEON"] = "_neon"
output_files["AVX2"] = "_avx2_gcc"
output_files["AVX-512+GFNI"] = "_avx512_gfni_gcc"

print(header)

for platform in ["NEON", "AVX2", "AVX-512+GFNI"]:

    print(f"\\multirow{{6}}{{*}}{{{platform}}}")

    for level in ["1","3","5"]:
        for variation in ["fast", "short"]:

            if (variation=="fast"):
                print(f"& \\multirow{{2}}{{*}}{{{roman[level]}}} & fast r3 &", end='')
            else:
                print("& & short r3 & ", end='')

            with open(f"{paths_opt[platform]}mqom2_cat{level}_gf16_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as optimized_rowf:
                conteudo_row_f = optimized_rowf.read()

            keygen_row = float(re.findall(r"^ - Key Gen: (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            sign_row = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            verify_row = float(re.findall(r"^ - Verify:  (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6

            with open(f"{paths_reference[platform]}mqom2_cat{level}_gf16_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as submissionf:
                conteudo_submission = submissionf.read()

            keygen_submission = float(re.findall(r"^ - Key Gen: (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            sign_submission = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            verify_submission = float(re.findall(r"^ - Verify:  (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6

            print(f" {keygen_row:.2f} & {keygen_submission:.2f} & {sign_row:.2f} & {sign_submission:.2f} & {verify_row:.2f} & {verify_submission:.2f} \\\\")
        if level != '5':
            print("\\addlinespace[1pt]")
    
    if platform == "AVX-512+GFNI":
        print("\\bottomrule")
    else:
        print("\\midrule")

print(footer)
