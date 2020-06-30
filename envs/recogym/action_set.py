import numpy as np

def get_rnd_action_sets(n_prods):
    # randomly split
    item_ids = [i for i in range(n_prods)]
    np.random.shuffle(item_ids)

    # split into half
    split_size = len(item_ids) // 2
    return item_ids[:split_size], item_ids[split_size:]

