SIGNS = {
    "0": "OK",
    "1": "THUMB_UP",
    "2": "TWO",
    "3": "THREE",
    "4": "SPIDERMAN",
    "5": "OPEN_HAND",
    "6": "FIST",
    "7": "PINCH",
    "8": "THUMB_DOWN",
    "9": "INDEX",
    "10": "MIDDLE",
    "11": "LITTLE"
}


def onehot(index: int, size: int = 16) -> list:
    list = [0 for i in range(size)]
    list[index] = 1
    return list
