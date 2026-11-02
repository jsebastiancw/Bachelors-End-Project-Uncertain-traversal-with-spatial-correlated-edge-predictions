from imports import *


def resolve_data_path(file_path: str) -> str:
    """
    Resolves dataset paths across `sevac/data/...` and legacy `Networks/...`
    so SEVAC runs standalone from any working directory.
    """
    if os.path.exists(file_path):
        return file_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sevac_root = os.path.abspath(os.path.join(base_dir, '..', '..'))

    # Normalize slashes and strip leading 'Networks/' if pointing to a network in sevac/data
    normalized = file_path.replace('//', '/').replace('\\', '/')
    if normalized.startswith('Networks/'):
        rel_part = normalized[len('Networks/'):]
    elif normalized.startswith('data/'):
        rel_part = normalized[len('data/'):]
    else:
        rel_part = normalized

    candidates = [
        os.path.join(base_dir, file_path),
        os.path.join(sevac_root, 'data', rel_part),
        os.path.join(sevac_root, normalized),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return file_path


def load_data(file_path: str) -> pd.DataFrame:
    resolved = resolve_data_path(file_path)
    return pd.read_csv(resolved)
