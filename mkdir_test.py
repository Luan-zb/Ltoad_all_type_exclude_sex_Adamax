import os
cur=0
results_dir="./results"
print('\nTraining Fold {}!'.format(cur))
writer_dir = os.path.join(results_dir, str(cur))
if not os.path.isdir(writer_dir):
    os.mkdir(writer_dir)
