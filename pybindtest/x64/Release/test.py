import pybindtest
from pybindtest import add, POINT

print('doc=', pybindtest.__doc__)
x1, y1 = 1, 2
print("add({},{})=".format(x1, y1), pybindtest.add(x1, y1))

x2, y2 = 3, 4

p = POINT(x1, y1)
q = POINT([x2, y2])

print('p=', p)
print('q=', q)
print('p+q=', p+q)
print('p.x,p.y=', p.x, p.y)