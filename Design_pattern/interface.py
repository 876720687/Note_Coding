from abc import ABCMeta,abstractmethod

# 接口
class Payment(metaclass=ABCMeta):
    # abstract class
    # 这个抽象方法并不需要实现,如果一个类里面有抽象方法，那么就是抽象类。
    @abstractmethod
    def pay(self, money):
        pass

# 上面的创建方式比下面的好处在于下面这种抽象类的创建当时只有当调用到了不符合规则的类对应的方法才会报错
# 上面的方法好处在于避免了存在抽象方法的情况，限制一定要按照抽象类的方式来编写，否则直接报错

# class Payment:
#     def pay(self, money):
#         raise NotImplementedError

# class Alipay(Payment):
#     pass

class Alipay(Payment):
    def pay(self, money):
        print("支付宝支付%d元."% money)

class WechatPay(Payment):
    def pay(self, money):
        print("微信支付%d元."% money)


p=Alipay()
p.pay(100)
q=WechatPay()
q.pay(1000)
# when you did not use it, it won't test the error

class User:
    def show_name(self):
        pass

class VIPUser(User):
    def show_name(self):
        pass

def show_user(u):
    res = u.show_name()




