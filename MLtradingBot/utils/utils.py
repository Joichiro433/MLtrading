def bool_from_str(text: str) -> bool:
    """文字列の真偽値をboolに変換する"""
    if text.lower() == 'true':
        return True
    if text.lower() == 'false':
        return False