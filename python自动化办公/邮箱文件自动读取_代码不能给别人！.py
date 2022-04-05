#-*- coding : utf-8-*-
# import poplib
# import email
# import email.header as EH
#
#
# """
#     需求：消息标题、附件名称（存在header中）都是以字节为单位进行传输的，中文内容需要解码
#     功能：对header进行解码
#     参考文档：
#         https://zhuanlan.zhihu.com/p/144172498
#         https://zhuanlan.zhihu.com/p/347516449
# """
# def decode(header: str):
#     value, charset = EH.decode_header(header)[0]
#     if charset:
#         return str(value, encoding=charset)
#     else:
#         return value
#
#
#
# """
#     功能：下载某一个消息的所有附件
# """
# def download_attachment(msg):
#     subject = decode(msg.get('Subject'))  # 获取消息标题
#     for part in msg.walk():  # 遍历整个msg的内容
#         if part.get_content_disposition() == 'attachment':
#             attachment_name = decode(part.get_filename())  # 获取附件名称
#             # attachment = attachment.encode("utf-8")
#             attachment_content = part.get_payload(decode=True)  # 下载附件
#             attachment_file = open('D:\\工具\\test_download\\' + attachment_name, 'wb') # 在指定目录下创建文件，注意二进制文件需要用wb模式打开
#             attachment_file.write(attachment_content)  # 将附件保存到本地
#             attachment_file.close()
#     print('Done………………', subject)
#
#
# def main():
#     """连接到POP3服务器"""
#     # server = poplib.POP3(host='pop.163.com')  # 创建一个POP3对象，参数host是指定服务器
#     #
#     # """身份验证"""
#     # server.user('yemenng8888@163.com')  # 参数是你的邮箱地址
#     # server.pass_('IEQJNCPHSEFMWYCH')  # 参数是你的邮箱密码，如果出现poplib.error_proto: b'-ERR login fail'，就用开启POP3服务时拿到的授权码
#
#     server = poplib.POP3(host='smtp.qq.com')
#     server.user('876720687@qq.com')
#     server.pass_('signaeeymshtbegc')
#
#     """获取邮箱中消息（邮件）数量"""
#     msg_count, _ = server.stat()
#
#     """遍历消息并保存附件"""
#     for i in range(msg_count):
#         """获取消息内容：POP3.retr(which)检索index为which的整个消息，并将其设为已读"""
#         _, lines, _ = server.retr(
#             i+1)  # 3个结果分别是响应结果（1个包含是否请求成功和该消息大小的字符串），消息内容（一个字符串列表，每个元素是消息内容的一行），消息大小（即有多少个octets，octet特指8bit的字节）
#
#         """将bytes格式的消息内容拼接"""
#         msg_bytes_content = b'\r\n'.join(lines)
#
#         """将字符串格式的消息内容转换为email模块支持的格式（<class 'email.message.Message'>）"""
#         msg = email.message_from_bytes(msg_bytes_content)
#
#         """下载消息中的附件"""
#         download_attachment(msg)
#
#
# if __name__ == "__main__":
#     main()


import poplib
import email
import time
from email.parser import Parser
from email.header import decode_header


def decode_str(s):#字符编码转换
    value, charset = decode_header(s)[0]
    if charset:
        value = value.decode(charset)
    return value


def get_att(msg):
    attachment_files = []

    for part in msg.walk():
        file_name = part.get_filename()  # 获取附件名称类型
        contType = part.get_content_type()

        if file_name:
            h = email.header.Header(file_name)
            dh = email.header.decode_header(h)  # 对附件名称进行解码
            filename = dh[0][0]
            if dh[0][1]:
                filename = decode_str(str(filename, dh[0][1]))  # 将附件名称可读化
                print(filename)
                # filename = filename.encode("utf-8")
            data = part.get_payload(decode=True)  # 下载附件
            att_file = open('D:\\保存代码\\' + filename, 'wb')  # 在指定目录下创建文件，注意二进制文件需要用wb模式打开
            attachment_files.append(filename)
            att_file.write(data)  # 保存附件
            att_file.close()
    return attachment_files



with open('D:\\config.txt', 'r') as f1:
    config = f1.readlines()
for i in range(0, len(config)):
    config[i] = config[i].rstrip('\n')

# print(config)

# POP3服务器、用户名、密码
host = config[0]  # pop.163.com
username = config[1]  # 用户名
password = config[2]  # 密码

# 连接到POP3服务器
server = poplib.POP3(host)

# 身份验证
server.user(username)
server.pass_(password) # 参数是你的邮箱密码，如果出现poplib.error_proto: b'-ERR login fail'，就用开启POP3服务时拿到的授权码

# stat()返回邮件数量和占用空间:
# print('Messages: %s. Size: %s' % server.stat())

# 可以查看返回的列表类似[b'1 82923', b'2 2184', ...]
resp, mails, octets = server.list()
# print(mails)

# 倒序遍历邮件
index = len(mails)
if index > 100:
    index = 100

for i in range(index, 0, -1):
    # lines存储了邮件的原始文本的每一行
    resp, lines, octets = server.retr(i)

    # 邮件的原始文本:
    msg_content = b'\r\n'.join(lines).decode('unicode_escape')

    # 解析邮件:
    msg = Parser().parsestr(msg_content)

    # 获取附件
    f_list = get_att(msg)

    print("附件获取完成")

print("文件已下载完成，10秒后关闭程序！")
time.sleep(10)