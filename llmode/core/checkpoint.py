"""Checkpoint utilities for saving and restoring experiment state."""
from __future__ import annotations

import os
import pickle
import time
from typing import Any

CHECKPOINT_FILE = 'checkpoint.pkl'


def save(log_dir: str, database: Any, global_sample_nums: int) -> None:
    """Atomically save experiment state to log_dir/checkpoint.pkl.

    If register_program has been monkey-patched with a local closure (which
    pickle cannot serialize), we temporarily restore the original bound method
    for the duration of pickling, then put the patch back.
    """
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, CHECKPOINT_FILE)
    tmp_path = path + '.tmp'

    # Remove any unpicklable monkey-patch before serializing.
    patched = database.__dict__.get('register_program')
    if patched is not None:
        del database.__dict__['register_program']

    try:
        state = {
            'database': database,
            'global_sample_nums': global_sample_nums,
            'saved_at': time.time(),
        }
        with open(tmp_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, path)  # atomic on POSIX
        print(f'[Checkpoint] Saved at sample {global_sample_nums} → {path}')
    finally:
        # Always restore the patch so the running pipeline is unaffected.
        if patched is not None:
            database.__dict__['register_program'] = patched


def load(log_dir: str):
    """Load checkpoint from log_dir.

    Returns:
        (database, global_sample_nums) if checkpoint exists, else (None, None).
    """
    path = os.path.join(log_dir, CHECKPOINT_FILE)
    if not os.path.exists(path):
        return None, None
    with open(path, 'rb') as f:
        state = pickle.load(f)
    gn = state.get('global_sample_nums', 1)
    saved_at = state.get('saved_at')
    ts = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(saved_at)) if saved_at else 'unknown'
    print(f'[Checkpoint] Resuming from sample {gn} (saved {ts}) ← {path}')
    return state['database'], gn
