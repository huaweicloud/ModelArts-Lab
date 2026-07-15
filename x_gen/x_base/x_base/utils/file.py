import os


def read_text(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


def list_cases(batch_test_input_dir: str):
    """
    返回 case_dir 列表：只取包含 image.jpg 与 prompt.txt 的子目录
    先过滤，再排序，避免 .DS_Store 等混入导致排序报错
    """
    case_names = []

    for name in os.listdir(batch_test_input_dir):
        case_dir = os.path.join(batch_test_input_dir, name)
        if not os.path.isdir(case_dir):
            continue

        img = os.path.join(case_dir, "image.jpg")
        txt = os.path.join(case_dir, "prompt.txt")
        if os.path.isfile(img) and os.path.isfile(txt):
            case_names.append(name)

    # 再排序：数字按数值排，其它按字符串排（稳）
    case_names = sorted(case_names, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))

    return [os.path.join(batch_test_input_dir, name) for name in case_names]
