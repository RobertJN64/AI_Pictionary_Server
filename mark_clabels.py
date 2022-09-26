import os

labels = sorted([name.removesuffix('.npy') for name in os.listdir('qd_files')])
not_labels = sorted([name.removesuffix('.npy') for name in os.listdir('qd_unused')])

with open("clabels.txt", 'a') as f:
    f.write("USED: ")
    f.write(', '.join(labels))
    f.write('\nUNUSED: ')
    f.write(', '.join(not_labels))
    f.write('\n\n')