header = r"""\begin{table}[H]\centering\color{black}\scriptsize
\renewcommand{\arraystretch}{1.00}\setlength{\tabcolsep}{2.15pt}
\begin{tabular}{@{}cll rr@{}}
\toprule
& & & \multicolumn{2}{c}{Peak dynamic memory (KB)} \\
\cmidrule(lr){4-5}
Platform & Level & Variant & Ours & Ref. \\
\midrule"""

footer = r"""\end{tabular}
\caption{Peak dynamic memory for MQOM v2.1 over $GF(2^4)$. Bold marks the lower (better) value in each Ours/Reference pair; ties are left unbolded.}\label{tab:memory_allocation}
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
paths_reference["AVX2/AVX-512"] = "./AVX/Reference/"
paths_opt["AVX2/AVX-512"] = "./AVX/Row-wise/"

output_files = {}
output_files["NEON"] = "_neon"
output_files["AVX2/AVX-512"] = "_avx2_gcc"

print(header)

for platform in ["NEON", "AVX2/AVX-512"]:

    print(f"\\multirow{{6}}{{*}}{{{platform}}}")

    for level in ["1","3","5"]:
        for variation in ["fast", "short"]:

            if (variation=="fast"):
                print(f"& \\multirow{{2}}{{*}}{{{roman[level]}}} & fast r3 &", end='')
            else:
                print("& & short r3 & ", end='')

            with open(f"{paths_opt[platform]}mqom2_cat{level}_gf16_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as optimized_rowf:
                conteudo_row_f = optimized_rowf.read()

            memory_row = int(re.findall(r"^\[\+\] Malloc stats keygen\+sign\+verif: Peak Usage is (.*) bytes$", conteudo_row_f, flags=re.MULTILINE)[0])/1e3

            with open(f"{paths_reference[platform]}mqom2_cat{level}_gf16_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as submissionf:
                conteudo_submission = submissionf.read()

            memory_submission = int(re.findall(r"^\[\+\] Malloc stats keygen\+sign\+verif: Peak Usage is (.*) bytes$", conteudo_submission, flags=re.MULTILINE)[0])/1e3


            print(f" {memory_row:.2f} & {memory_submission:.2f} \\\\")
        if level != '5':
            print("\\addlinespace[1pt]")

    if platform == "AVX2/AVX-512":
        print("\\bottomrule")
    else:
        print("\\midrule")

print(footer)



