import argparse
import os
import orjson

from unstructured.cleaners.core import clean_bullets,clean_dashes,clean_extra_whitespace,clean_non_ascii_chars,clean_ordered_bullets,clean_postfix,clean_prefix,clean_trailing_punctuation,group_broken_paragraphs,replace_unicode_quotes

cleaning_functions = {
    'bullets': clean_bullets,
    'dashes': clean_dashes,
    'extra_whitespace': clean_extra_whitespace,
    'non_ascii_chars': clean_non_ascii_chars,
    'ordered_bullets': clean_ordered_bullets,
    'trailing_punctuation': clean_trailing_punctuation,
    'group_broken_paragraphs': group_broken_paragraphs,
    'replace_unicode_quotes': replace_unicode_quotes
}

def clean_in_order(text,functions=None):
    if functions is None:
        functions = cleaning_functions.keys()
    for func in functions:
        if func in cleaning_functions:
            text = cleaning_functions[func](text)
    return text

def clean_corpus(input_file, output_file, id_field, content_field,
                 clean_functions=[
                    'bullets',
                    'dashes',
                    'extra_whitespace',
                    'ordered_bullets',
                    'group_broken_paragraphs',
                    'replace_unicode_quotes'
                 ]):
    import colorama
    print(colorama.Fore.GREEN + f"cleaning {input_file} to {output_file}" + colorama.Style.RESET_ALL)
    output_diff = []
    with open(input_file, 'r',encoding='UTF-8') as f:
        for line in f:
            data_obj = orjson.loads(line)
            id = data_obj[id_field]
            content = data_obj[content_field]
            cleaned_content = clean_in_order(content,clean_functions)
            cleaned_obj = {}
            cleaned_obj[id_field] = id
            cleaned_obj['prev_content'] = content
            if cleaned_content != content:
                cleaned_obj[content_field] = cleaned_content
                output_diff.append(
                    cleaned_obj
                )
    print(colorama.Fore.RED+ f"cleaned {len(output_diff)} out of {len(data_obj)}" + colorama.Style.RESET_ALL)
    print(colorama.Fore.GREEN + f"saving to {output_file}" + colorama.Style.RESET_ALL)
    with open(output_file, 'w',encoding='UTF-8') as f:
        for obj in output_diff:
            f.write(orjson.dumps(obj).decode('utf-8')+'\n')
    print(colorama.Fore.GREEN + f"done writing to {output_file}" + colorama.Style.RESET_ALL)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='clean corpus for training')
    parser.add_argument('--input', type=str, help='input file or directory')
    parser.add_argument('--output', type=str, help='output file or directory')
    parser.add_argument('--id_field', type=str,default='ID', help='field to use as id')
    parser.add_argument('--content_field', type=str,default='Content', help='field to use as text')
    parser.add_argument('--num_processes', type=int, default=4, help='number of processes to use')

    args = parser.parse_args()
    if args.input is None or args.output is None:
        parser.print_help()
        exit(1)

    print(f"cleaning {args.input} to {args.output}")

    is_dir = True

    if os.path.isfile(args.input):    
        is_dir = False

    if is_dir and os.path.isfile(args.output):
        print(f"output {args.output} should be a directory because input {args.input} is a directory")
        exit(1)
    if is_dir :
        os.makedirs(args.output, exist_ok=True)
    else:
        #delete output file if it exists
        if os.path.isfile(args.output):
            os.remove(args.output)
    if is_dir:
        import multiprocessing as mp
        with mp.Pool(args.num_processes) as pool:
            for file in os.listdir(args.input):
                if file.endswith('.json') or file.endswith('.jsonl'):
                    pool.apply_async(
                        clean_corpus,
                        args=(os.path.join(args.input,file),os.path.join(args.output,file),args.id_field,args.content_field)
                    )
                    # clean_corpus(
                    #     os.path.join(args.input,file),
                    #     os.path.join(args.output,file),
                    #     args.id_field,
                    #     args.content_field
                    # )
            pool.close()
            pool.join()
    else:
        clean_corpus(
            args.input,
            args.output,
            args.id_field,
            args.content_field
        )

    
    
    
    