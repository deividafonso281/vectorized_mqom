header = r"""\begin{table}[H]\centering\scriptsize
\renewcommand{\arraystretch}{1.00}\setlength{\tabcolsep}{2.15pt}
\begin{tabular}{@{}cll rr rr rr rr@{}}
\toprule
& & & \multicolumn{2}{c}{Encryption} & \multicolumn{2}{c}{XOF} & \multicolumn{2}{c}{MatrixMult} & \multicolumn{2}{c}{Fold} \\
\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}
Platform & Level & Variant & Ours & Ref. & Ours & Ref. & Ours & Ref. & Ours & Ref. \\
\midrule"""

footer = r"""\end{tabular}
\caption{Signature-generation profile for MQOM v2.1 over $GF(2^4)$, in millions of cycles. Bold marks the lower (better) value in each Ours/Reference pair; ties are left unbolded.}\label{tab:profileGF16combined}
\end{table}"""

from pathlib import Path
import re

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

    for level in ["1", "3", "5"]:
        for variation in ["fast", "short"]:

            if (variation=="fast"):
                print(f"& \\multirow{{2}}{{*}}{{{roman[level]}}} & fast r3 &", end='')
            else:
                print("& & short r3 & ", end='')

            with open(f"{paths_reference[platform]}mqom2_cat{level}_gf16_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as submissionf:
                conteudo_submission = submissionf.read()

            with open(f"{paths_opt[platform]}mqom2_cat{level}_gf16_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as rowf:
                conteudo_row = rowf.read()

            expand_trees_submission = float(re.findall(r"^   - \[BLC.Commit\] Expand Trees:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            seed_commit_submission = float(re.findall(r"^   - \[BLC.Commit\] Seed Commit:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            prg_submission = float(re.findall(r"^   - \[BLC.Commit\] PRG:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            xof_submission = float(re.findall(r"^   - \[BLC.Commit\] XOF:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            arithm_submission = float(re.findall(r"^   - \[BLC.Commit\] Arithm:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            matrix_mul_ext_submission = float(re.findall(r"^   - \[PIOP.Compute\] Matrix Mul Ext:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            sample_challenge_submission = float(re.findall(r"^   - Sample Challenge:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            sign_submission = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6

            expand_trees_row = float(re.findall(r"^   - \[BLC.Commit\] Expand Trees:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            seed_commit_row = float(re.findall(r"^   - \[BLC.Commit\] Seed Commit:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            prg_row = float(re.findall(r"^   - \[BLC.Commit\] PRG:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            xof_row = float(re.findall(r"^   - \[BLC.Commit\] XOF:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            arithm_row = float(re.findall(r"^   - \[BLC.Commit\] Arithm:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            matrix_mul_ext_row = float(re.findall(r"^   - \[PIOP.Compute\] Matrix Mul Ext:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            sample_challenge_row = float(re.findall(r"^   - Sample Challenge:.*\((\d+(?:\.\d+)?)\s+cycles\)", conteudo_row, flags=re.MULTILINE)[0])/1e6
            sign_row = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_row, flags=re.MULTILINE)[0])/1e6

            shake_submission = xof_submission + sample_challenge_submission
            enc_submission = expand_trees_submission + seed_commit_submission + prg_submission 

            shake_row = xof_row + sample_challenge_row
            enc_row = expand_trees_row + seed_commit_row + prg_row 

            print(f" {enc_row:.2f} & {enc_submission:.2f} & {shake_row:.2f} & {shake_submission:.2f} & {matrix_mul_ext_row:.2f} & {matrix_mul_ext_submission:.2f} & {arithm_row:.2f}  & {arithm_submission:.2f}\\\\")
        if level != '5':
            print("\\addlinespace[1pt]")
    
    if platform == "AVX-512+GFNI":
        print("\\bottomrule")
    else:
        print("\\midrule")

print(footer)
