class Person():
    def __init__(self,name,age,gener="男"):
        self.name=name
        self.age=age
        self.gener=gener
    def _del_(self):
        print("Bye bye——from",self.name)
    def printInfo(self):
        print("姓名:",self.name,"年龄:",self.age,"性别:",self.gener)

zhansan=Person("张三",18)
lisi=Person("李四",19,"女")

zhansan.printInfo()
lisi.printInfo()