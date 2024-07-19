from enum import Enum, auto, unique
from pathlib import Path
import warnings

# 一時的に警告を無視する
warnings.filterwarnings("ignore", category=DeprecationWarning)

''' ================================================================= '''
# 特定のパスを1要素のリストで取得
def get_path(mattress):
    '''
    args:
        matress : (Enum)type.tester.mattress
    '''
    return [mattress.value]

''' ================================================================= '''
# tester用のスーパークラス
class tester(Enum):
    # 全要素のパスを取得
    @classmethod
    def all(cls):
        e = []
        for value in cls.__members__.values():
            e.append(value.value)
        return e

    # 全要素のパスを確認
    @classmethod
    def check_item(cls):
        e = cls.all()
        for i in e:
            print(i)

    # fl のマットレスの情報のみを取得
    @classmethod
    def fl(cls):
        e = []
        attr_M = False
        try:
            e.append(cls.fl_center.value)
        except AttributeError:
            attr_M = True

        try:
            if attr_M:
                e.append(cls.fls_center.value)
                e.append(cls.fld_center.value)
                e.append(cls.flf_center.value)
        except AttributeError:
            print("this tester does not have fl data")
            exit(1)

        return e

''' ================================================================= '''
# テスター用のスーパークラス
class type(tester):
    @classmethod
    def all(cls):
        e = []
        for v in cls.__members__.values():
            e.extend(v.value.all())
        return e

    @classmethod
    def serch(cls, name):
        e = []
        match(name):
            case 'fl':
                for v in cls.__members__.values():
                    e.extend(v.value.fl())
            case 'st':
                try:
                    for v in cls.__members__.values():
                        e.append(v.value.st_center.value)
                except AttributeError:
                    print("AttributeError : this type does not have st data")
                    exit(1)
            case 'ka':
                for v in cls.__members__.values():
                    e.append(v.value.ka_center.value)
                    e.append(v.value.ka_right.value)
                    e.append(v.value.ka_left.value)
            case _:
                pass
        return e

''' ================================================================= '''
# pathを一元管理するEnumクラス
class type_LMH(type):
    class tester_H002(tester):
        fl_center = Path("raw/LMH/H002/fl_center")
        st_center = Path("raw/LMH/H002/st_center")
        ka_center = Path("raw/LMH/H002/ka_center")
        ka_right = Path("raw/LMH/H002/ka_right")
        ka_left = Path("raw/LMH/H002/ka_left")

    class tester_H003(tester):
        fl_center = Path("raw/LMH/H003/fl_center")
        st_center = Path("raw/LMH/H003/st_center")
        ka_center = Path("raw/LMH/H003/ka_center")
        ka_right = Path("raw/LMH/H003/ka_right")
        ka_left = Path("raw/LMH/H003/ka_left")

    class tester_L001(tester):
        fl_center = Path("raw/LMH/L001/fl_center")
        st_center = Path("raw/LMH/L001/st_center")
        ka_center = Path("raw/LMH/L001/ka_center")
        ka_right = Path("raw/LMH/L001/ka_right")
        ka_left = Path("raw/LMH/L001/ka_left")

    class tester_L003(tester):
        fl_center = Path("raw/LMH/L003/fl_center")
        st_center = Path("raw/LMH/L003/st_center")
        ka_center = Path("raw/LMH/L003/ka_center")
        ka_right = Path("raw/LMH/L003/ka_right")
        ka_left = Path("raw/LMH/L003/ka_left")

    class tester_M001(tester):
        fls_center = Path("raw/LMH/M001/fls_center")
        fld_center = Path("raw/LMH/M001/fld_center")
        flf_center = Path("raw/LMH/M001/fls_center")
        st_center = Path("raw/LMH/M001/st_center")
        ka_center = Path("raw/LMH/M001/ka_center")
        ka_right = Path("raw/LMH/M001/ka_right")
        ka_left = Path("raw/LMH/M001/ka_left")

    class tester_M002(tester):
        fls_center = Path("raw/LMH/M002/fls_center")
        fld_center = Path("raw/LMH/M002/fld_center")
        flf_center = Path("raw/LMH/M002/fls_center")
        st_center = Path("raw/LMH/M002/st_center")
        ka_center = Path("raw/LMH/M002/ka_center")
        ka_right = Path("raw/LMH/M002/ka_right")
        ka_left = Path("raw/LMH/M002/ka_left")

    class tester_M003(tester):
        fls_center = Path("raw/LMH/M003/fls_center")
        fld_center = Path("raw/LMH/M003/fld_center")
        flf_center = Path("raw/LMH/M003/fls_center")
        st_center = Path("raw/LMH/M003/st_center")
        ka_center = Path("raw/LMH/M003/ka_center")
        ka_right = Path("raw/LMH/M003/ka_right")
        ka_left = Path("raw/LMH/M003/ka_left")

    class tester_M004(tester):
        fls_center = Path("raw/LMH/M004/fls_center")
        fld_center = Path("raw/LMH/M004/fld_center")
        flf_center = Path("raw/LMH/M004/fls_center")
        st_center = Path("raw/LMH/M004/st_center")
        ka_center = Path("raw/LMH/M004/ka_center")
        ka_right = Path("raw/LMH/M004/ka_right")
        ka_left = Path("raw/LMH/M004/ka_left")

    H002 = tester_H002
    H003 = tester_H003
    L001 = tester_L001
    L003 = tester_L003
    M001 = tester_M001
    M002 = tester_M002
    M003 = tester_M003
    M004 = tester_M004

class type_YMGT(type):
    class tester_YMGT_1(tester):
        ka_center = Path("raw/YMGT/YMGT_1/ka_center")
        ka_right = Path("raw/YMGT/YMGT_1/ka_right")
        ka_left = Path("raw/YMGT/YMGT_1/ka_left")

    class tester_YMGT_2(tester):
        ka_center = Path("raw/YMGT/YMGT_2/ka_center")
        ka_right = Path("raw/YMGT/YMGT_2/ka_right")
        ka_left = Path("raw/YMGT/YMGT_2/ka_left")

    class tester_YMGT_3(tester):
        ka_center = Path("raw/YMGT/YMGT_3/ka_center")
        ka_right = Path("raw/YMGT/YMGT_3/ka_right")
        ka_left = Path("raw/YMGT/YMGT_3/ka_left")

    class tester_YMGT_4(tester):
        ka_center = Path("raw/YMGT/YMGT_4/ka_center")
        ka_right = Path("raw/YMGT/YMGT_4/ka_right")
        ka_left = Path("raw/YMGT/YMGT_4/ka_left")

    class tester_YMGT_5(tester):
        ka_center = Path("raw/YMGT/YMGT_5/ka_center")
        ka_right = Path("raw/YMGT/YMGT_5/ka_right")
        ka_left = Path("raw/YMGT/YMGT_5/ka_left")

    class tester_YMGT_6(tester):
        ka_center = Path("raw/YMGT/YMGT_6/ka_center")
        ka_right = Path("raw/YMGT/YMGT_6/ka_right")
        ka_left = Path("raw/YMGT/YMGT_6/ka_left")

    class tester_YMGT_7(tester):
        ka_center = Path("raw/YMGT/YMGT_7/ka_center")
        ka_right = Path("raw/YMGT/YMGT_7/ka_right")
        ka_left = Path("raw/YMGT/YMGT_7/ka_left")

    class tester_YMGT_8(tester):
        ka_center = Path("raw/YMGT/YMGT_8/ka_center")
        ka_right = Path("raw/YMGT/YMGT_8/ka_right")
        ka_left = Path("raw/YMGT/YMGT_8/ka_left")

    class tester_YMGT_9(tester):
        ka_center = Path("raw/YMGT/YMGT_9/ka_center")
        ka_right = Path("raw/YMGT/YMGT_9/ka_right")
        ka_left = Path("raw/YMGT/YMGT_9/ka_left")

    class tester_YMGT_10(tester):
        ka_center = Path("raw/YMGT/YMGT_10/ka_center")
        ka_right = Path("raw/YMGT/YMGT_10/ka_right")
        ka_left = Path("raw/YMGT/YMGT_10/ka_left")

    class tester_YMGT_11(tester):
        ka_center = Path("raw/YMGT/YMGT_11/ka_center")
        ka_right = Path("raw/YMGT/YMGT_11/ka_right")
        ka_left = Path("raw/YMGT/YMGT_11/ka_left")

    YMGT1 = tester_YMGT_1
    YMGT2 = tester_YMGT_2
    YMGT3 = tester_YMGT_3
    YMGT4 = tester_YMGT_4
    YMGT5 = tester_YMGT_5
    YMGT6 = tester_YMGT_6
    YMGT7 = tester_YMGT_7
    YMGT8 = tester_YMGT_8
    YMGT9 = tester_YMGT_9
    YMGT10 = tester_YMGT_10
    YMGT11 = tester_YMGT_11

class type_YMGT2023(type):
    class tester_YMGT_1(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_1/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_1/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_1/ka_left")

    class tester_YMGT_2(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_2/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_2/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_2/ka_left")

    class tester_YMGT_3(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_3/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_3/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_3/ka_left")

    class tester_YMGT_4(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_4/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_4/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_4/ka_left")

    class tester_YMGT_5(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_5/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_5/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_5/ka_left")

    class tester_YMGT_6(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_6/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_6/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_6/ka_left")

    class tester_YMGT_7(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_7/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_7/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_7/ka_left")

    class tester_YMGT_8(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_8/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_8/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_8/ka_left")

    class tester_YMGT_9(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_9/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_9/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_9/ka_left")

    class tester_YMGT_10(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_10/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_10/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_10/ka_left")

    class tester_YMGT_11(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_11/ka_center")
        ka_right = Path("raw/YMGT_2023/YMGT_11/ka_right")
        ka_left = Path("raw/YMGT_2023/YMGT_11/ka_left")

    YMGT1 = tester_YMGT_1
    YMGT2 = tester_YMGT_2
    YMGT3 = tester_YMGT_3
    YMGT4 = tester_YMGT_4
    YMGT5 = tester_YMGT_5
    YMGT6 = tester_YMGT_6
    YMGT7 = tester_YMGT_7
    YMGT8 = tester_YMGT_8
    YMGT9 = tester_YMGT_9
    YMGT10 = tester_YMGT_10
    YMGT11 = tester_YMGT_11

LMH = type_LMH
YMGT = type_YMGT
YMGT2023 = type_YMGT2023

e = LMH.H002.value.fl_center
print(e.name)