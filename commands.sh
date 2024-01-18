pip install -r "requirements.txt"
python Pipeline/dst.py 0 > "Pipeline/Results/output_0.txt"
python Pipeline/dst.py 1 > "Pipeline/Results/output_1.txt"
python Pipeline/dst.py 2 > "Pipeline/Results/output_2.txt"
python Pipeline/dst.py 3 > "Pipeline/Results/output_3.txt"
python cleanup.py 4