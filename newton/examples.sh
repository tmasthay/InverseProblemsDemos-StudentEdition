python newton_test.py output_file=newton
python newton_test.py method=BFGS output_file=BFGS
python newton_test.py method=Nelder-Mead output_file=NelderMead
python newton_test.py method=Powell output_file=Powell

cpdf all_cases.pdf *.pdf
