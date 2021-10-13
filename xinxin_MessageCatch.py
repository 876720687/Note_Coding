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

# POP3服务器、用户名、密码
host = config[0]  # pop.163.com
username = config[1]  # 用户名
password = config[2]  # 密码

# 连接到POP3服务器
server = poplib.POP3(host)

# 身份验证
server.user(username)
server.pass_(password) # 参数是你的邮箱密码，如果出现poplib.error_proto: b'-ERR login fail'，就用开启POP3服务时拿到的授权码

resp, mails, octets = server.list()

index = len(mails)
# if index > 50:
#     index = 50

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