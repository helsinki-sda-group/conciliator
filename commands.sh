pip install -r "requirements.txt"
# Priority profiles
# 1. "eco": 1/10 treasure, 2/10 time and 7/10 fuel
# 2. "gold digger": 98/100 treasure, 1/100 time and 1/100 fuel
# 3. "balanced": 1/5 treasure, 2/5 time and 2/5 fuel
python3 Pipeline/dst.py 0 > Pipeline/Results/output_0.txt
python3 Pipeline/dst.py 1 > Pipeline/Results/output_1.txt
python3 Pipeline/dst.py 2 > Pipeline/Results/output_2.txt
python3 Pipeline/cleanup.py 3