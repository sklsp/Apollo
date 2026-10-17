import os

from .core import run_agent
from .io import import_document, list_supported_files


def main():
    default_path = 'data/company_document.txt'
    document = ''
    if os.path.exists(default_path):
        document = import_document(default_path)

    print('AI Document Agent — interactive')
    print('Commands: load <path>, summary, keywords, files, exit')

    while True:
        text = input('\nAsk something (or command): ').strip()
        if not text:
            continue
        parts = text.split(maxsplit=1)
        op = parts[0].lower()

        if op == 'exit':
            break
        if op == 'load' and len(parts) > 1:
            path = parts[1].strip()
            if not os.path.exists(path):
                print('File not found:', path)
                continue
            document = import_document(path)
            print(f'Loaded document from {path} ({len(document)} chars)')
            continue
        if op == 'summary':
            if not document:
                print('No document loaded.')
                continue
            print(run_agent('summary', document))
            continue
        if op == 'keywords':
            if not document:
                print('No document loaded.')
                continue
            print(run_agent('keyword', document))
            continue
        if op == 'files':
            print('\n'.join(list_supported_files()))
            continue

        if not document:
            print('No document loaded. Use "load <path>" or place files in data/')
            continue

        try:
            print(run_agent(text, document))
        except Exception as exc:
            print(f'Error: {exc}')


if __name__ == '__main__':
    main()
