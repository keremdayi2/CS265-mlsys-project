from dataclasses import dataclass, fields

def obj_list_to_array(lst):
    if len(lst) == 0:
        return []
    
    keys = [field.name for field in fields(lst[0])]

    ret = [keys]

    for l in lst:
        row = []
        for key in keys:
            val = getattr(l, key)
            row.append(val)
        
        ret.append(row)
    
    return ret