from functools import reduce
import numpy as np

texts, pos = ['3', 'A', 'B', 'U', '0', 'O'], [2, 5, 8, 11, 13, 14]
texts2, pos2 = ['3', 'C', 'A', '5', 'W', 'M', 'A', '0', '1', '7', '0'], [2, 5, 8, 11, 16, 20, 24, 27, 31, 34, 36]

def filter(texts, pos):
    arr = np.array(pos)
    diff = np.diff(arr)
    print(np.diff(arr) < np.floor(np.average(np.diff(arr))))
    ignore_index = []
    for i, coincidental in enumerate(diff < np.floor(np.average(diff))):
        confuse_letters = ["0", "O", "o"]
        if coincidental:
            if texts[i] in confuse_letters and texts[i+1] in confuse_letters:
                ignore_index.append(i+1)
    new_texts, new_pos = [], []
    for i in range(len(texts)):
        if i in ignore_index:
            continue
        new_texts.append(texts[i])
        new_pos.append(pos[i])
    return new_texts, new_pos

print(filter(texts, pos))
print(filter(texts2, pos2))
