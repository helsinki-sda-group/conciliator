import pandas as pd
import os
import sys
import numpy as np

# Number of outputs
j = 3
if len(sys.argv) > 1:
    j = int(sys.argv[1])

# Emissions data cleanup
frames = []
for i in range(j):
    df = pd.read_csv(f"Pipeline/Results/emissions_{i}.csv")
    frames.append(df)
    os.remove(f"Pipeline/Results/emissions_{i}.csv")

all_results = pd.concat(frames).reset_index(drop=True)
all_results.to_csv("Pipeline/Results/all_emissions.csv")
temp = all_results.apply(pd.to_numeric, errors = "coerce").mean()
avgs = pd.DataFrame(data=np.array([temp]), columns = list(all_results.columns)).reset_index(drop=True)
final_results = avgs.fillna(all_results).reset_index(drop=True)
final_results.to_csv("Pipeline/Results/avg_emissions.csv")

# Output cleanup
delimiter = "\n\n" + "="*20 + "\n\n"
with open('Pipeline/Results/whole_output.txt', 'a') as outfile:
    for i in range(j):
        filename = f"Pipeline/Results/output_{i}.txt"
        with open(filename, 'r') as infile:
            outfile.write(infile.read())
            outfile.write(delimiter)
        os.remove(filename)