import pandas as pd
import os
import sys

# Emissions data cleanup
frames = []
for i in range(4):
    df = pd.read_csv(f"Pipeline/Results/emissions_{i}.csv")
    frames.append(df)

total_results = pd.concat(frames)
avg_results = pd.concat(frames)
avg_results.apply(pd.to_numeric, errors = "ignore")
avgs = avg_results.mean(axis=0, numeric_only=True)
print(avg_results.head())
final_results = avg_results.combine_first(total_results)
print(final_results.head())

# Output cleanup
# filename = f'/Results/output_{0}.txt'
# with open('/Results/output.txt', 'w') as outfile:
#     for i in range(4):
#         for file in filename.format(i):
#             with open(file, 'r') as infile:
#                 outfile.write(infile.read())
#                 outfile.write("\n")

#for i in range(4):
#    for file in filename.format(i):
#        os.remove(file)