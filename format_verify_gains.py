header = r"""\begin{tikzpicture}[
    x=0.138cm,
    y=0.027cm,
    font=\small
]

\def\barw{1.15}
\def\sep{0.32}

% Soft panel backgrounds for visual grouping.
\fill[black!2] (1.5,-31) rectangle (34.2,75);
\fill[black!2] (35.8,-31) rectangle (68.5,75);

% Panel headings and subtle rules.
\node[font=\small\bfseries] at (17.5,88){Fast};
\node[font=\small\bfseries] at (52.2,88){Short};
\draw[black!20,line width=.4pt] (8,78) -- (27,78);
\draw[black!20,line width=.4pt] (42.5,78) -- (61.5,78);

% Horizontal guides. Keep the 0%% line visually dominant.
\foreach \y in {-10,15,30}{
    \draw[densely dashed,black!18,line width=.35pt] (0,2*\y) -- (70,2*\y);
}
\draw[black,line width=.75pt] (0,0) -- (70,0);
\draw[black!25,line width=.35pt] (35,-31) -- (35,75);

% y-axis and labels.
\draw[black!65,line width=.45pt] (0,-31) -- (0,65);
\foreach \y/\lab in {-10/-10\%,0/0\%,15/15\%,30/30\%}{
    \draw[black!55,line width=.35pt] (-0.25,2*\y)--(0,2*\y);
    \node[left=3pt,text=black!75] at (-0.2,2*\y){\lab};
}

\newcommand{\ThreeBars}[5]{
    \filldraw[fill=gf2,draw=black!55,line width=.28pt]
      (#1,0) rectangle ++(\barw,2*#2);
    \filldraw[fill=gf16,draw=black!55,line width=.28pt]
      ({#1+\barw+\sep},0) rectangle ++(\barw,2*#3);
    \filldraw[fill=gf256,draw=black!55,line width=.28pt]
      ({#1+2*(\barw+\sep)},0) rectangle ++(\barw,2*#4);
    \node[below=7pt,font=\small] at ({#1+\barw+\sep},0){#5};
}"""

footer = r"""% Compact centered legend.
\begin{scope}[shift={(18,-54)}]
  \filldraw[fill=gf2,draw=black!45,line width=.25pt] (0,0) rectangle +(2.1,8);
  \node[right=3pt] at (2.1,4){$GF(2)$};
  \filldraw[fill=gf16,draw=black!45,line width=.25pt] (14,0) rectangle +(2.1,8);
  \node[right=3pt] at (16.1,4){$GF(2^4)$};
  \filldraw[fill=gf256,draw=black!45,line width=.25pt] (29,0) rectangle +(2.1,8);
  \node[right=3pt] at (31.1,4){$GF(2^8)$};
\end{scope}

\end{tikzpicture}
"""

from pathlib import Path
import re

root = Path(".")

fields = {}
fields["gf2"] = "$GF(2)$"
fields["gf16"] = "$GF(2^4)$"
fields["gf256"] = "$GF(2^8)$"

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

start = -5

for variation in ["fast", "short"]:

    for platform in ["NEON", "AVX2", "AVX-512+GFNI"]:
        start = start + 10
        print(f"\\ThreeBars{{{start}}}",end="")
        for field in ["gf2","gf16","gf256"]:

            with open(f"{paths_opt[platform]}mqom2_cat1_{field}_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as optimized_rowf:
                conteudo_row_f = optimized_rowf.read()

            keygen_row = float(re.findall(r"^ - Key Gen: (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            sign_row = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6
            verify_row = float(re.findall(r"^ - Verify:  (.*) cycles$", conteudo_row_f, flags=re.MULTILINE)[0])/1e6

            with open(f"{paths_reference[platform]}mqom2_cat1_{field}_{variation}_r3/bench{output_files[platform]}.txt", "r", encoding="utf-8") as submissionf:
                conteudo_submission = submissionf.read()

            keygen_submission = float(re.findall(r"^ - Key Gen: (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            sign_submission = float(re.findall(r"^ - Sign:    (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6
            verify_submission = float(re.findall(r"^ - Verify:  (.*) cycles$", conteudo_submission, flags=re.MULTILINE)[0])/1e6

            
            print(f"{{{100*(verify_submission-verify_row)/verify_submission:.2f}}}",end="")
        print(f"{{{platform}}}")
    
    start = start + 5

print(footer)