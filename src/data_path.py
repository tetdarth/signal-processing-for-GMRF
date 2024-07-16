from enum import Enum
from pathlib import Path

class mattresses(Enum):
    fl_center = "fl_center"
    st_center = "st_center"
    ka_center = "ka_center"
    ka_right = "ka_right"
    ka_left = "ka_left"

class testers(Enum):
    H002 = "H002"
    H003 = "H003"
    L001 = "L001"
    L003 = "L003"
    M001 = "M001"
    M002 = "M002"
    M003 = "M003"
    M004 = "M004"
    YMGT1 = "YMGT_1"
    YMGT2 = "YMGT_2"
    YMGT3 = "YMGT_3"
    YMGT4 = "YMGT_4"
    YMGT5 = "YMGT_5"
    YMGT6 = "YMGT_6"
    YMGT7 = "YMGT_7"
    YMGT8 = "YMGT_8"
    YMGT9 = "YMGT_9"
    YMGT10 = "YMGT_10"
    YMGT11 = "YMGT_11"

class type(Enum):
    LMH = "LMH"
    YMGT = "YMGT"
    YMGT_2023 = "YMGT_2023"


# ファイルパスの取得
def get_path(_type, _tester, _mattress):
    p = [Path("raw" , f"{_type.value}/{_tester.value}")]
    if _tester == testers.M001 or _tester == testers.M002 or _tester == testers.M003 or _tester == testers.M004:
        if _mattress == mattresses.fl_center:
            p = [Path(p[0], "fls_center"), Path(p[0], "fld_center"), Path(p[0], "flf_center")]
        else:
            p = [Path(p[0], f"{_mattress.value}")]
    else:
        p = [Path(p[0], f"{_mattress.value}")]

    # 適切なファイルが存在しない場合の例外処理
    try:
        if not p[0].exists():
            raise FileNotFoundError("The specified directory does not exist.")
    except FileNotFoundError as e:
        print(e)
        print("file not found...")
        if _type == type.YMGT or _type == type.YMGT_2023:
            if _mattress == mattresses.fl_center or _mattress == mattresses.st_center:
                print(f"tester.{_tester.value} has the following attributes :")
                print("ka_center\nka_left\nka_right\n ...")
            print(f"type.{_type.value} has the following attributes :")
            print("YMGT1\nYMGT2\nYMGT3\n ...")
        else:
            print(f"type.{_type.value} has the following attributes :")
            print("H002\nH003\nL001\n ...")
        exit(1)

    return p