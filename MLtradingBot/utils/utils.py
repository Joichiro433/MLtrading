def bool_from_str(text: str) -> bool:
    """文字列の真偽値をboolに変換する"""
    if text.lower() == 'true':
        return True
    if text.lower() == 'false':
        return False

def round_num(number: float, base: float = 0.5) -> float:
    """numberをbase刻みで丸める"""
    round_base = 1 / base
    return round(number * round_base) / round_base