PHAA_SPLIT_NUM = None
ENABLE_PHAA = False


def set_phaa_split_num(phaa_split_num: int):
    global PHAA_SPLIT_NUM
    PHAA_SPLIT_NUM = phaa_split_num


def get_phaa_split_num() -> int:
    return PHAA_SPLIT_NUM


def enable_phaa():
    global ENABLE_PHAA
    ENABLE_PHAA = True


def is_phaa_enabled() -> bool:
    return ENABLE_PHAA
