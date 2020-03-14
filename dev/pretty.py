def pretty(d, indent=4, pre_indent=0):
    for key, value in d.items():
        print((indent * " ") * pre_indent + str(key))
        if isinstance(value, dict):
            pretty(value, pre_indent=pre_indent + 1)
        else:
            print((indent * " ") * (pre_indent + 1) + str(value) + "\n")
