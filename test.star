msg = "Hi there!"

def foo():
  def bar(a):
    return a
  return bar

print("Message:", foo()(msg))

x = 10 - 3
print(x)

y = 5 + 2
print(y)

z = 10 + 5 - 3 * 2 + 1
print("The result is:", z)
