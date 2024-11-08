from enum import Enum, auto, unique
from pathlib import Path
import warnings
import re

# 一時的に警告を無視する
warnings.filterwarnings("ignore", category=DeprecationWarning)

''' ================================================================= '''
# 特定のパスを1要素のリストで取得
def get_path(mattress):
    '''
    args:
        matress : type.tester.(value).mattress
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
            e.append(value)
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
                if cls == LMH.M004.value:
                    e.append(cls.fls_center.value)
                    e.append(cls.fls_left.value)
                    e.append(cls.fls_right.value)
                else:
                    e.append(cls.fls_center.value)
                    e.append(cls.fld_center.value)
                    e.append(cls.flf_center.value)
        except AttributeError:
            print("this tester does not have fl data")
            assert(1)

        return e

    @classmethod
    def to_str(cls):
        match(cls.__name__):
            case 'tester_H001':
                return 'H001'
            case 'tester_H002':
                return 'H002'
            case 'tester_H003':
                return 'H003'
            case 'tester_L001':
                return 'L001'
            case 'tester_L003':
                return 'L003'
            case 'tester_M001':
                return 'M001'
            case 'tester_M002':
                return 'M002'
            case 'tester_M003':
                return 'M003'
            case 'tester_M004':
                return 'M004'
            case _:
                pass
        return cls.__name__

''' ================================================================= '''
# type用のスーパークラス
class type(tester):
    @classmethod
    def all(cls):
        e = []
        for v in cls.__members__.values():
            e.extend(v.value.all())
        return list(set(e))

    @classmethod
    def serch(cls, name, skip=[]):
        e = []
        match(name):
            case 'fl':
                for v in cls.__members__.values():
                    flag = False
                    for s in skip:
                        if s == v:
                            flag = True
                    if flag:
                        continue
                    e.extend(v.value.fl())
                    
            case 'st':
                try:
                    for v in cls.__members__.values():
                        flag = False
                        for s in skip:
                            if s == v:
                                flag = True
                        if flag:
                            continue
                        e.append(v.value.st_center.value)
                except AttributeError:
                    print("AttributeError : this type does not have st data")
                    exit(1)
                    
            case 'ka':
                for v in cls.__members__.values():
                    flag = False
                    for s in skip:
                        if s == v:
                            flag = True
                    if flag:
                        continue
                    e.append(v.value.ka_center.value)
                    e.append(v.value.ka_right.value)
                    e.append(v.value.ka_left.value)
                    
            case _:
                pass
        e = set(e)
        return list(e)

    @classmethod
    def to_str(cls):
        match(cls.__name__):
            case 'type_LMH':
                return 'LMH'
            case 'type_YMGT':
                return 'YMGT'
            case 'type_YMGT2023':
                return 'YMGT2023'
            case _:
                pass

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
        flf_center = Path("raw/LMH/M001/flf_center")
        st_center = Path("raw/LMH/M001/st_center")
        ka_center = Path("raw/LMH/M001/ka_center")
        ka_right = Path("raw/LMH/M001/ka_right")
        ka_left = Path("raw/LMH/M001/ka_left")

    class tester_M002(tester):
        fls_center = Path("raw/LMH/M002/fls_center")
        fld_center = Path("raw/LMH/M002/fld_center")
        flf_center = Path("raw/LMH/M002/flf_center")
        st_center = Path("raw/LMH/M002/st_center")
        ka_center = Path("raw/LMH/M002/ka_center")
        ka_right = Path("raw/LMH/M002/ka_right")
        ka_left = Path("raw/LMH/M002/ka_left")

    class tester_M003(tester):
        fls_center = Path("raw/LMH/M003/fls_center")
        fld_center = Path("raw/LMH/M003/fld_center")
        flf_center = Path("raw/LMH/M003/flf_center")
        st_center = Path("raw/LMH/M003/st_center")
        ka_center = Path("raw/LMH/M003/ka_center")
        ka_right = Path("raw/LMH/M003/ka_right")
        ka_left = Path("raw/LMH/M003/ka_left")

    class tester_M004(tester):
        fls_center = Path("raw/LMH/M004/fls_center")
        fls_left = Path("raw/LMH/M004/fls_left")
        fls_right = Path("raw/LMH/M004/fls_right")
        st_center = Path("raw/LMH/M004/st_center")
        st_right = Path("raw/LMH/M004/st_right")
        st_left = Path("raw/LMH/M004/st_left")
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
        ka_center = Path("raw/YMGT_2023/YMGT_1/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_1/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_1/Air_left")

    class tester_YMGT_2(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_2/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_2/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_2/Air_left")

    class tester_YMGT_3(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_3/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_3/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_3/Air_left")

    class tester_YMGT_5(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_5/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_5/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_5/Air_left")

    class tester_YMGT_6(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_6/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_6/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_6/Air_left")

    class tester_YMGT_7(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_7/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_7/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_7/Air_left")

    class tester_YMGT_8(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_8/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_8/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_8/Air_left")

    class tester_YMGT_9(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_9/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_9/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_9/Air_left")

    class tester_YMGT_10(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_10/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_10/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_10/Air_left")

    class tester_YMGT_11(tester):
        ka_center = Path("raw/YMGT_2023/YMGT_11/Air_center")
        ka_right = Path("raw/YMGT_2023/YMGT_11/Air_right")
        ka_left = Path("raw/YMGT_2023/YMGT_11/Air_left")

    YMGT1 = tester_YMGT_1
    YMGT2 = tester_YMGT_2
    YMGT3 = tester_YMGT_3
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

def mattress_all(position=False):
    if position == False:
        return ['ka', 'st', 'fl']
    else:
        return ['ka_center', 'ka_right', 'ka_left', 'fl_center', 'fl_right', 'fl_left', 'st_center', 'fld_center', 'flf_center', 'fls_center', 'fls_left', 'fls_right']

def parent(espec):
    match(espec):
        # LMH._
        case LMH.H002.value.fl_center | LMH.H002.value.st_center | LMH.H002.value.ka_center | LMH.H002.value.ka_left | LMH.H002.value.ka_right:
            return LMH.H002
        case LMH.H003.value.fl_center | LMH.H003.value.st_center | LMH.H003.value.ka_center | LMH.H003.value.ka_left | LMH.H003.value.ka_right:
            return LMH.H003
        case LMH.L001.value.fl_center | LMH.L001.value.st_center | LMH.L001.value.ka_center | LMH.L001.value.ka_left | LMH.L001.value.ka_right:
            return LMH.L001
        case LMH.L003.value.fl_center | LMH.L003.value.st_center | LMH.L003.value.ka_center | LMH.L003.value.ka_left | LMH.L003.value.ka_right:
            return LMH.L003
        case LMH.M001.value.fls_center | LMH.M001.value.fld_center | LMH.M001.value.flf_center | LMH.M001.value.st_center | LMH.M001.value.ka_center | LMH.M001.value.ka_left | LMH.M001.value.ka_right:
            return LMH.M001
        case LMH.M002.value.fls_center | LMH.M002.value.fld_center | LMH.M002.value.flf_center | LMH.M002.value.st_center | LMH.M002.value.ka_center | LMH.M002.value.ka_left | LMH.M002.value.ka_right:
            return LMH.M002
        case LMH.M003.value.fls_center | LMH.M003.value.fld_center | LMH.M003.value.flf_center | LMH.M003.value.st_center | LMH.M003.value.ka_center | LMH.M003.value.ka_left | LMH.M003.value.ka_right:
            return LMH.M003
        case LMH.M004.value.fls_center | LMH.M004.value.fls_left | LMH.M004.value.fls_right | LMH.M004.value.st_center | LMH.M004.value.st_left | LMH.M004.value.st_right | LMH.M004.value.ka_center | LMH.M004.value.ka_left | LMH.M004.value.ka_right:
            return LMH.M004

        # YMGT._
        case YMGT.YMGT1.value.ka_center | YMGT.YMGT1.value.ka_left | YMGT.YMGT1.value.ka_right:
            return YMGT.YMGT1
        case YMGT.YMGT2.value.ka_center | YMGT.YMGT2.value.ka_left | YMGT.YMGT2.value.ka_right:
            return YMGT.YMGT2
        case YMGT.YMGT3.value.ka_center | YMGT.YMGT3.value.ka_left | YMGT.YMGT3.value.ka_right:
            return YMGT.YMGT3
        case YMGT.YMGT4.value.ka_center | YMGT.YMGT4.value.ka_left | YMGT.YMGT4.value.ka_right:
            return YMGT.YMGT4
        case YMGT.YMGT5.value.ka_center | YMGT.YMGT5.value.ka_left | YMGT.YMGT5.value.ka_right:
            return YMGT.YMGT5
        case YMGT.YMGT6.value.ka_center | YMGT.YMGT6.value.ka_left | YMGT.YMGT6.value.ka_right:
            return YMGT.YMGT6
        case YMGT.YMGT7.value.ka_center | YMGT.YMGT7.value.ka_left | YMGT.YMGT7.value.ka_right:
            return YMGT.YMGT7
        case YMGT.YMGT8.value.ka_center | YMGT.YMGT8.value.ka_left | YMGT.YMGT8.value.ka_right:
            return YMGT.YMGT8
        case YMGT.YMGT9.value.ka_center | YMGT.YMGT9.value.ka_left | YMGT.YMGT9.value.ka_right:
            return YMGT.YMGT9
        case YMGT.YMGT10.value.ka_center | YMGT.YMGT10.value.ka_left | YMGT.YMGT10.value.ka_right:
            return YMGT.YMGT10
        case YMGT.YMGT11.value.ka_center | YMGT.YMGT11.value.ka_left | YMGT.YMGT11.value.ka_right:
            return YMGT.YMGT11

        # YMGT2023._
        case YMGT2023.YMGT1.value.ka_center | YMGT2023.YMGT1.value.ka_left | YMGT2023.YMGT1.value.ka_right:
            return YMGT2023.YMGT1
        case YMGT2023.YMGT2.value.ka_center | YMGT2023.YMGT2.value.ka_left | YMGT2023.YMGT2.value.ka_right:
            return YMGT2023.YMGT2
        case YMGT2023.YMGT3.value.ka_center | YMGT2023.YMGT3.value.ka_left | YMGT2023.YMGT3.value.ka_right:
            return YMGT2023.YMGT3
        case YMGT2023.YMGT5.value.ka_center | YMGT2023.YMGT5.value.ka_left | YMGT2023.YMGT5.value.ka_right:
            return YMGT2023.YMGT5
        case YMGT2023.YMGT6.value.ka_center | YMGT2023.YMGT6.value.ka_left | YMGT2023.YMGT6.value.ka_right:
            return YMGT2023.YMGT6
        case YMGT2023.YMGT7.value.ka_center | YMGT2023.YMGT7.value.ka_left | YMGT2023.YMGT7.value.ka_right:
            return YMGT2023.YMGT7
        case YMGT2023.YMGT8.value.ka_center | YMGT2023.YMGT8.value.ka_left | YMGT2023.YMGT8.value.ka_right:
            return YMGT2023.YMGT8
        case YMGT2023.YMGT9.value.ka_center | YMGT2023.YMGT9.value.ka_left | YMGT2023.YMGT9.value.ka_right:
            return YMGT2023.YMGT9
        case YMGT2023.YMGT10.value.ka_center | YMGT2023.YMGT10.value.ka_left | YMGT2023.YMGT10.value.ka_right:
            return YMGT2023.YMGT10
        case YMGT2023.YMGT11.value.ka_center | YMGT2023.YMGT11.value.ka_left | YMGT2023.YMGT11.value.ka_right:
            return YMGT2023.YMGT11

        # type
        case LMH.H002 | LMH.H003 | LMH.L001 | LMH.L003 | LMH.M001 | LMH.M002 | LMH.M003 | LMH.M004:
            return LMH
        case YMGT.YMGT1 | YMGT.YMGT2 | YMGT.YMGT3 | YMGT.YMGT4 | YMGT.YMGT5 | YMGT.YMGT6 | YMGT.YMGT7 | YMGT.YMGT8 | YMGT.YMGT9 | YMGT.YMGT10 | YMGT.YMGT11:
            return YMGT
        case YMGT2023.YMGT1 | YMGT2023.YMGT2 | YMGT2023.YMGT3 | YMGT2023.YMGT5 | YMGT2023.YMGT6 | YMGT2023.YMGT7 | YMGT2023.YMGT8 | YMGT2023.YMGT9 | YMGT2023.YMGT10 | YMGT2023.YMGT11:
            return YMGT2023

    # 対応するパスが見つからない場合は None を返す
    return None

def getattributes(identifier, include_position = False):
    type = parent(parent(identifier)).to_str()
    tester = parent(identifier).value.to_str()

    def get_mattress():
        match identifier:

            case (LMH.H002.value.ka_center | LMH.H003.value.ka_center | LMH.L001.value.ka_center |
                LMH.L003.value.ka_center | LMH.M001.value.ka_center | LMH.M002.value.ka_center |
                LMH.M003.value.ka_center | LMH.M004.value.ka_center |
                YMGT.YMGT1.value.ka_center | YMGT.YMGT2.value.ka_center | YMGT.YMGT3.value.ka_center |
                YMGT.YMGT4.value.ka_center | YMGT.YMGT5.value.ka_center | YMGT.YMGT6.value.ka_center |
                YMGT.YMGT7.value.ka_center | YMGT.YMGT8.value.ka_center | YMGT.YMGT9.value.ka_center |
                YMGT.YMGT10.value.ka_center | YMGT.YMGT11.value.ka_center |
                YMGT2023.YMGT1.value.ka_center | YMGT2023.YMGT2.value.ka_center | YMGT2023.YMGT3.value.ka_center |
                YMGT2023.YMGT5.value.ka_center | YMGT2023.YMGT6.value.ka_center |
                YMGT2023.YMGT7.value.ka_center | YMGT2023.YMGT8.value.ka_center | YMGT2023.YMGT9.value.ka_center |
                YMGT2023.YMGT10.value.ka_center | YMGT2023.YMGT11.value.ka_center):
                return 'ka_center'

            case (LMH.H002.value.ka_left | LMH.H003.value.ka_left | LMH.L001.value.ka_left |
                LMH.L003.value.ka_left | LMH.M001.value.ka_left | LMH.M002.value.ka_left |
                LMH.M003.value.ka_left | LMH.M004.value.ka_left |
                YMGT.YMGT1.value.ka_left | YMGT.YMGT2.value.ka_left | YMGT.YMGT3.value.ka_left |
                YMGT.YMGT4.value.ka_left | YMGT.YMGT5.value.ka_left | YMGT.YMGT6.value.ka_left |
                YMGT.YMGT7.value.ka_left | YMGT.YMGT8.value.ka_left | YMGT.YMGT9.value.ka_left |
                YMGT.YMGT10.value.ka_left | YMGT.YMGT11.value.ka_left |
                YMGT2023.YMGT1.value.ka_left | YMGT2023.YMGT2.value.ka_left | YMGT2023.YMGT3.value.ka_left |
                YMGT2023.YMGT5.value.ka_left | YMGT2023.YMGT6.value.ka_left |
                YMGT2023.YMGT7.value.ka_left | YMGT2023.YMGT8.value.ka_left | YMGT2023.YMGT9.value.ka_left |
                YMGT2023.YMGT10.value.ka_left | YMGT2023.YMGT11.value.ka_left):
                return 'ka_left'

            case (LMH.H002.value.ka_right | LMH.H003.value.ka_right | LMH.L001.value.ka_right |
                LMH.L003.value.ka_right | LMH.M001.value.ka_right | LMH.M002.value.ka_right |
                LMH.M003.value.ka_right | LMH.M004.value.ka_right |
                YMGT.YMGT1.value.ka_right | YMGT.YMGT2.value.ka_right | YMGT.YMGT3.value.ka_right |
                YMGT.YMGT4.value.ka_right | YMGT.YMGT5.value.ka_right | YMGT.YMGT6.value.ka_right |
                YMGT.YMGT7.value.ka_right | YMGT.YMGT8.value.ka_right | YMGT.YMGT9.value.ka_right |
                YMGT.YMGT10.value.ka_right | YMGT.YMGT11.value.ka_right |
                YMGT2023.YMGT1.value.ka_right | YMGT2023.YMGT2.value.ka_right | YMGT2023.YMGT3.value.ka_right |
                YMGT2023.YMGT5.value.ka_right | YMGT2023.YMGT6.value.ka_right |
                YMGT2023.YMGT7.value.ka_right | YMGT2023.YMGT8.value.ka_right | YMGT2023.YMGT9.value.ka_right |
                YMGT2023.YMGT10.value.ka_right | YMGT2023.YMGT11.value.ka_right):
                return 'ka_right'

            case (LMH.H002.value.fl_center | LMH.H003.value.fl_center | LMH.L001.value.fl_center |
                LMH.L003.value.fl_center
            ):
                return 'fl_center'

            case (LMH.H002.value.st_center | LMH.H003.value.st_center | LMH.L001.value.st_center |
                LMH.L003.value.st_center | LMH.M001.value.st_center | LMH.M002.value.st_center |
                LMH.M003.value.st_center | LMH.M004.value.st_center
            ):
                return 'st_center'
            case LMH.M004.value.st_left:
                return 'st_left'
            case LMH.M004.value.st_right:
                return 'st_right'
            case LMH.M001.value.fld_center | LMH.M002.value.fld_center | LMH.M003.value.fld_center:
                return 'fld_center'
            case LMH.M001.value.flf_center | LMH.M002.value.flf_center | LMH.M003.value.flf_center:
                return 'flf_center'
            case LMH.M001.value.fls_center | LMH.M002.value.fls_center | LMH.M003.value.fls_center | LMH.M004.value.fls_center:
                return 'fls_center'
            case LMH.M004.value.fls_left:
                return 'fls_left'
            case LMH.M004.value.fls_right:
                return 'fls_right'

    mattress = get_mattress()
    position = ''
    if include_position == False:
        match(mattress):
            case 'fl_center' | 'fld_center' | 'flf_center' | 'fls_center' | 'ka_center' | 'st_center':
                position = 'center'
            case 'fls_left' | 'ka_left' | 'st_left':
                position = 'left'
            case 'fls_right' | 'ka_right' | 'st_right':
                position = 'right'
            case _:
                assert()

        match(mattress):
            case 'fl_center' | 'fld_center' | 'flf_center' | 'fls_center' | 'fls_left' | 'fls_right':
                mattress = 'fl'
            case 'ka_center' | 'ka_left' | 'ka_right':
                mattress = 'ka'
            case 'st_center' | 'st_left' | 'st_right':
                mattress ='st'
            case _:
                assert()

    return type, tester, mattress, position

def extract_position(paths, position):
    e = []
    for p in paths:
        if re.search(position, str(p)) is not None:
            e.append(p)
    e = set(e)
    return list(e)


# e = LMH.H003.value.ka_center
'''
for i in LMH.serch('fl', skip=[LMH.M004]):
    print(i)
'''

'''
for i in extract_position(LMH.all(), "center"):
    print(i)
'''
# print(LMH.L001.value.all())