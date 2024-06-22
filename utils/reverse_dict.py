def reverse_dict(x: dict):
    """交换字典的 key-value, 得到 value-key 的新字典
    需保证value无重复项
    """
    if isinstance(x, dict):
        k, v = list(zip(*list(x.items())))
        x_reverse = {}
        for i in range(len(k)):
            x_reverse[v[i]] = k[i]  # k-v 反转字典
        return x_reverse
    else:
        raise TypeError('arg needs to be dict')