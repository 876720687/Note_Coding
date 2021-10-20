# coding: utf-8
# 创建一个类，类名称第一个字母大写,可以带括号也可以不带括号
# python中同样使用关键字class创建一个类，类名称第一个字母大写,可以带括号也可以不带括号；
# python中实例化类不需要使用关键字new（也没有这个关键字），类的实例化类似函数调用方式；


class Student():
    student_count = 0
    def __init__(self, name, salary):
        self.name = name
        self.age = salary
        Student.student_count += 1
    def display_count(self):
        print('Total student {}'.format(Student.student_count))
    def display_student(self):
        print('Name: {}, age: {}'.format(self.name,self.age))
    def get_class(self):
        if self.age >= 7 and self.age < 8:
            return 1
        if self.age >= 8 and self.age < 9:
            return 2
        if self.age >= 9 and self.age < 10:
            return 3
        if self.age >= 10 and self.age < 11:
            return 4
        else:
            return  0


student1 = Student('cuiyongyuan', 10)
student2 = Student('yuanli', 10)

student1.display_student()
student2.display_student()

student1_class = student1.get_class()
student2_class = student2.get_class()



