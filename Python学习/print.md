### 基本语法
```
print(*objects, sep=' ', end='\n', file=sys.stdout, flush=False)
```
sep为object有多个内容时，不同object间的分隔，而end为多次输出指令间的间隔

```
lst=('a','b','c')
[In]
print(lst)
[Out]
('a', 'b', 'c')

[In]
for item in lst:
	print(item,end='')
[Out]
abc

[In]
print(*lst)
[Out]
a b c

[In]
print(*lst,sep='')
[Out]
abc
```
此外有print(* lst) 这种特殊格式，在Python中，print(* b)是一种特殊的语法，称为解包（unpacking）操作。这里的 * 符号用于解包列表b，这意味着它将列表b中的每个元素作为单独的参数传递给print函数。
当你调用print( * b)时，Python会将列表b展开成多个参数，然后将这些参数传递给print函数。这与直接调用print(b)是不同的，因为print(b)会将整个列表作为一个参数传递给print函数，这通常会导致输出列表的字符串表示形式，包括方括号和逗号

可见 for item的形式实际上是执行了多次print操作
而 * lst 的操作实际上是将字符串中多个参数给print，而进行一次print操作

### 输出格式
为规定小数位数，可采用’%.2f’%f，采用的是四舍五入法
```
f = 2.3456789

print('%.2f'%f)
print('%.3f'%f)
print('%.4f'%f)
```
这种方法同时也可以进行函数赋值

