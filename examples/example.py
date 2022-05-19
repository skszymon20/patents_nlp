def train(model: int) -> int:
    """_summary_

    Args:
        model (int): _description_

    Returns:
        int: _description_
    """
    if model > 0:
        return 1
    else:
        return 0


if __name__ == '__main__':
    print(train(2))
    # to test pycodestyle `$pycodestyle example.py`
    # to test mypy `$mypy example.py`
