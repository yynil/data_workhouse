if __name__ == '__main__':
    from unstructured.partition.auto import partition
    input_file = "/media/yueyulin/TOUROS/tmp/2402.05608v2.pdf"
    elements = partition(input_file)
    print("\n\n".join([str(el) for el in elements]))