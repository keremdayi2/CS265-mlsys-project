from dataclasses import dataclass, fields

def obj_list_to_array(lst):
    if len(lst) == 0:
        return []

    keys = list(filter(lambda x: x[:2] != '__' and x[-2:] != '__', dir(lst[0])))

    ret = [keys]

    for l in lst:
        row = []
        for key in keys:
            val = getattr(l, key)
            row.append(val)
        
        ret.append(row)
    
    return ret