from imports import *


def resolve_data_path(file_path: str) -> str:
    """
    Resolves dataset paths against the unified root `networks/` directory
    so SEVAC runs standalone from any working directory.
    """
    if os.path.exists(file_path):
        return file_path

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, '..', '..', '..'))

    # Normalize slashes and strip leading 'networks/', 'Networks/', or 'data/'
    normalized = file_path.replace('//', '/').replace('\\', '/')
    for prefix in ('../networks/', 'networks/', 'Networks/', 'data/'):
        if normalized.startswith(prefix):
            rel_part = normalized[len(prefix):]
            break
    else:
        rel_part = normalized

    candidates = [
        os.path.join(base_dir, file_path),
        os.path.join(repo_root, 'networks', rel_part),
        os.path.join(repo_root, normalized),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate

    return file_path


def load_data(file_path: str) -> pd.DataFrame:
    resolved = resolve_data_path(file_path)
    return pd.read_csv(resolved)
