"""
https://www.jianshu.com/p/cc8a4dbc370c
"""

# 第一个简略的模拟
import hashlib
import datetime

class DaDaBlockCoin:

    #index 索引，timestamp 时间戳，data 交易记录，self_hash交易hash,last_hash,上个hash
    def __init__(self,idex,timestamp,data,last_hash):
        self.idex = idex
        self.timestamp = timestamp
        self.data = data
        self.last_hash = last_hash
        self.self_hash=self.hash_DaDaBlockCoin()


    def hash_DaDaBlockCoin(self):
        sha = hashlib.md5()#加密算法,这里可以选择sha256,sha512,为了打印方便，所以选了md5
        #对数据整体加密
        datastr = str(self.idex)+str(self.timestamp)+str(self.data)+str(self.last_hash)
        sha.update(datastr.encode("utf-8"))
        return sha.hexdigest()

def create_first_DaDaBlock():  # 创世区块

    return DaDaBlockCoin(0, datetime.datetime.now(), "love dadacoin", "0")


# last_block,上一个区块
def create_money_DadaBlock(last_block):  # 其它块
    this_idex = last_block.idex + 1  # 索引加1
    this_timestamp = datetime.datetime.now()
    this_data = "love dada" + str(this_idex)  # 模拟交易数据
    this_hash = last_block.self_hash  # 取得上一块的hash
    return DaDaBlockCoin(this_idex, this_timestamp, this_data, this_hash)


DaDaBlockCoins = [create_first_DaDaBlock()]  # 区块链列表，只有一个创世区块
nums = 10
head_block = DaDaBlockCoins[0]
print(head_block.idex, head_block.timestamp, head_block.self_hash, head_block.last_hash)
for i in range(nums):
    dadaBlock_add = create_money_DadaBlock(head_block)  # 创建一个区块链的节点
    DaDaBlockCoins.append(dadaBlock_add)
    head_block = dadaBlock_add
    print(dadaBlock_add.idex, dadaBlock_add.timestamp, dadaBlock_add.self_hash, dadaBlock_add.last_hash)












import hashlib
import json
import requests
from textwrap import dedent
from time import time
from uuid import uuid4
from urllib.parse import urlparse
from flask import Flask, jsonify, request


class Blockchain(object):
    def __init__(self):
        ...
        self.nodes = set()
        # 用 set 来储存节点，避免重复添加节点.
        ...
        self.chain = []
        self.current_transactions = []

        # 创建创世区块
        self.new_block(previous_hash=1, proof=100)

    def reister_node(self, address):
        """
        在节点列表中添加一个新节点
        :param address:
        :return:
        """
        prsed_url = urlparse(address)
        self.nodes.add(prsed_url.netloc)

    def valid_chain(self, chain):
        """
        确定一个给定的区块链是否有效
        :param chain:
        :return:
        """
        last_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(f'{last_block}')
            print(f'{block}')
            print("\n______\n")
            # 检查block的散列是否正确
            if block['previous_hash'] != self.hash(last_block):
                return False
            # 检查工作证明是否正确
            if not self.valid_proof(last_block['proof'], block['proof']):
                return False

            last_block = block
            current_index += 1
        return True


    def ressolve_conflicts(self):
        """
        共识算法
        :return:
        """
        neighbours = self.nodes
        new_chain = None
        # 寻找最长链条
        max_length = len(self.chain)

        # 获取并验证网络中的所有节点的链
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                # 检查长度是否长，链是否有效
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        # 如果发现一个新的有效链比当前的长，就替换当前的链
        if new_chain:
            self.chain = new_chain
            return True
        return False

    def new_block(self,proof,previous_hash=None):
        """
        创建一个新的块并将其添加到链中
        :param proof: 由工作证明算法生成证明
        :param previous_hash: 前一个区块的hash值
        :return: 新区块
        """
        block = {
            'index':len(self.chain)+1,
            'timestamp':time(),
            'transactions':self.current_transactions,
            'proof':proof,
            'previous_hash':previous_hash or self.hash(self.chain[-1]),
        }

        # 重置当前交易记录
        self.current_transactions = []

        self.chain.append(block)
        return block

    def new_transaction(self,sender,recipient,amount):
        # 将新事务添加到事务列表中
        """
        Creates a new transaction to go into the next mined Block
        :param sender:发送方的地址
        :param recipient:收信人地址
        :param amount:数量
        :return:保存该事务的块的索引
        """
        self.current_transactions.append({
            'sender':sender,
            'recipient':recipient,
            'amount':amount,
        })

        return self.last_block['index'] + 1


    @staticmethod # 静态方法无需实例化
    def hash(block):
        """
        给一个区块生成 SHA-256 值
        :param block:
        :return:
        """
        # 必须确保这个字典（区块）是经过排序的，否则将会得到不一致的散列
        block_string = json.dumps(block,sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property # 只能通过getter创建对象的时候进行赋值，无法重新赋值，保证了区块链信息的唯一性
    def last_block(self):
        # 返回链中的最后一个块
        return self.chain[-1]


    def proof_of_work(self,last_proof):
        # 工作算法的简单证明
        proof = 0
        while self.valid_proof(last_proof,proof)is False:
            proof +=1
        return proof

    @staticmethod
    def valid_proof(last_proof,proof):
        # 验证证明
        guess =  f'{last_proof}{proof}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] =="0000"


# 实例化节点，接收一个参数__name__
app = Flask(__name__)

# 为该节点生成一个全局惟一的地址
node_identifier = str(uuid4()).replace('-','')

# 实例化Blockchain类
blockchain = Blockchain()


# 进行挖矿请求，装饰器的作用是将路由映射到视图函数
@app.route('/mine',methods=['GET'])
def mine():
    # 运行工作算法的证明来获得下一个证明。
    last_block = blockchain.last_block
    last_proof = last_block['proof']
    proof = blockchain.proof_of_work(last_proof)

    # 必须得到一份寻找证据的奖赏。
    blockchain.new_transaction(
        sender="0",
        recipient=node_identifier,
        amount=1,
    )

    # 通过将其添加到链中来构建新的块
    previous_hash = blockchain.hash(last_block)
    block = blockchain.new_block(proof,previous_hash)
    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'previous_hash': block['previous_hash'],
    }
    return jsonify(response), 200


# 创建交易请求
@app.route('/transactions/new',methods=['POST'])
def new_transactions():
    values = request.get_json()

    # 检查所需要的字段是否位于POST的data中
    required = ['sender','recipient','amount']
    if not all(k in values for k in required):
        return 'Missing values',400

    #创建一个新的事物
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])
    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201


# 获取所有快信息
@app.route('/chain',methods=['GET'])
def full_chain():
    response = {
        'chain':blockchain.chain,
        'length':len(blockchain.chain),
    }
    return jsonify(response),200


# 添加节点
@app.route('/nodes/register',methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return "Error: Please supply a valid list of nodes", 400

    for node in nodes:
        blockchain.reister_node(node)

    response = {
        'message': 'New nodes have been added',
        'total_nodes': list(blockchain.nodes),
    }
    return jsonify(response), 201


# 解决冲突(解决冲突的函数并没有定义)
@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.ressolve_conflicts()

    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }

    return jsonify(response), 200

if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)





