def f(x):
    print("A", end="")
    if (x >= 0):
        print("B", end="")
        print("C", end="")
        #will print B and C if x greater than 0
    print("D") 
    #will print A and D no matter what

f(0)
f(1)