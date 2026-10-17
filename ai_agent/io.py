import os


def import_document(file_path):
    with open(file_path, encoding='utf-8') as f:
        return f.read()


def list_supported_files(directory='data'):
    if not os.path.exists(directory):
        return []
    return [p for p in os.listdir(directory) if os.path.isfile(os.path.join(directory, p))]
