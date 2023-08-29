def _flatten(T, N, value):
    """Flatten the first two dimensions of a tensor."""
    return value.reshape(T * N, *value.shape[2:])
