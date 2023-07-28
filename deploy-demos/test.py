import os
import requests
import json
import threading
import time
import tqdm
import numpy as np
import functools
from multiprocessing.dummy import Pool


def check_available_cuda_devices():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU| grep Used > logs/tmp_cuda")
    select_devices = []
    with open("logs/tmp_cuda") as f:
        for idx, line in enumerate(f):
            fields =line.split()
            print(fields)
            if int(fields[2]) < 1000:
                select_devices.append(idx)
    return select_devices        

def test_legacy_service():
    free_cuda_devices = check_available_cuda_devices()
    print(free_cuda_devices)

    data = {
        "model_id": "your-org/bloomS2",
        "query": "hi, you", "chatbot": [],
        "max_length": 512, "decoder": "sample", "top_p": 0.7,
        "repetition_penalty": 1.0,
        "temperature": 0.95, "history": []
    }
    headers={
        'Content-Type': 'application/json'
    }
    modelPORT=57601 #53101
    modelHost= "11.70.129.225" #"11.167.162.21" #"11.70.128.85"
    
    def genResponse(task_size = 1, batch_size = 1):
        cost_records =[]
        results = []
        # data['history'] = [("介绍一下蓝牙耳机的功能", "好的")]
        # data['query'] = "Show me the functionality of Bluetooth in 100 words."
        data['query'] = " ".join(1* ["介绍一下蓝牙耳机的功能"])
        # data['query'] = "根据用户对“监控摄像”的需求描述，预测其商品偏好列表，列表不超过10个偏好。\n用户肯定性需求是表达了需要某个偏好的需求；用户否定性需求是用“不需要”、“不要”、“没有”、“不带”、“不含”等否定词表达的需求。\n一般用户需求都会包含肯定性需求，很少的会包含否定性需求。\n如果某个商品偏好描述一旦包含在了用户的某个否定性需求中，则该偏好是“否定偏好”。\n如果某个商品偏好描述满足了至少一个用户肯定性需求且不是“否定偏好”，则该偏好是“肯定偏好”。\n不要对用户需求没有体现的偏好做任何判断。\n\n\n用户需求是：\n用户需要了解小米智能摄像机云台版2K的防水情况，以及是否有室外防水的摄像机推荐链接。\n\n商品偏好描述列表：\n#1:监控摄像 包装清单::小米智能摄像\n#2:监控摄像 特点::小米智能摄像机\n#3:监控摄像 特点::小米智能摄像机2\n#4:监控摄像 特点::智能摄像机云台版\n#5:监控摄像 特点::小米摄像头\n#6:监控摄像 特点::摄像机云台2K版\n#7:监控摄像 特点::米家智能\n#8:监控摄像 特点::云台2K版\n#9:监控摄像 特点::云台版SE+\n#10:监控摄像 特点::云台PRO\n#11:监控摄像 类型::家用云台\n#12:监控摄像 品牌::小米（MI）\n#13:监控摄像 特点::米家\n#14:监控摄像 特点::防尘防水\n#15:监控摄像 特点::云台版\n#16:监控摄像 特点::双云台\n#17:监控摄像 特点::防水\n#18:监控摄像 特点::云台\n#19:监控摄像 防水等级::不防水\n#20:监控摄像 特点::智能看家\n#21:监控摄像 特点::AI智能看家\n#22:监控摄像 特点::小米智能猫眼1\n#23:监控摄像 特点::智能摄像机\n#24:监控摄像 包装清单::XIAOM\n#25:监控摄像 特点::2K款\n#26:监控摄像 特点::标准版2K\n#27:监控摄像 特点::2K高清\n#28:监控摄像 特点::AI智能\n#29:监控摄像 特点::智能\n#30:监控摄像 特点::2K画质\n#31:监控摄像 防水等级::IPX5\n#32:监控摄像 特点::2K超高清\n#33:监控摄像 特点::2K超清\n#34:监控摄像 特点::2K分辨率\n#35:监控摄像 特点::入门款\n#36:监控摄像 特点::旗舰款\n#37:监控摄像 特点::2K\n#38:监控摄像 夜视类型::红外夜视\n#39:监控摄像 APP控制::水平+垂\n#40:监控摄像 特点::AI人形侦测\n#41:监控摄像 存储方式::云存\n#42:监控摄像 云台旋转角度::360\n#43:监控摄像 特点::室外新品\n#44:监控摄像 特点::室内小巧\n#45:监控摄像 夜视类型::全彩夜视\n#46:监控摄像 云台旋转角度::0度\n#47:监控摄像 特点::摄像头室外版\n#48:监控摄像 特点::防盗门监控\n#49:监控摄像 特点::WIFI\n#50:监控摄像 适用场景::居家室外\n\n"
        # "现在你需要扮演一个商品导购员，你需要根据用户的需求和每个商品的描述进行推荐。\n\n\n输入：\n\n用户需求: \n价格在2000-3000之间，支持公交卡刷卡功能的手机。\n\n商品列表:\n#1: Redmi 10A 5000mAh大电量 1300万AI相机 指纹解锁 6GB+128GB 暗影黑  小米 红米 合约机 购机补贴版  10A 6GB+128G 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-899.0\n#2: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 G88芯片 5000mAh电池 4GB+128GB 神秘黑境 手机 小米 红米【购机 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-999.0\n#3: Redmi 9A 5000mAh 1300万AI相机 人脸解锁 4GB+128GB 砂石黑 游戏 小米 红米 合约机 购机补贴版 超长续航 1300万像素 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-799.0\n#4: Redmi 9A 5000mAh大电量 1300万AI相机 人脸解锁 6GB+128GB 砂石黑 合约机 购机补贴版 高性能处理器 高清分辨率 长效续航 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-899.0\n#5: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 5000mAh电池 6GB+128GB 梦幻晴空 手机 小米 红米【直播】  梦幻晴 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1099.0\n#6: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 5000mAh电池 4GB+128GB 梦幻晴空 手机 小米 红米【直播】  梦幻晴 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-999.0\n#7: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 5000mAh电池 4GB+128GB 时光独白 手机 小米 红米【直播】 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-999.0\n#8: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 5000mAh电池 6GB+128GB 时光独白 手机 小米 红米【直播】  时光独 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1099.0\n#9: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 5000mAh电池 6GB+128GB 神秘黑境 手机 小米 红米【直播】  神秘黑 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1099.0\n#10: Redmi Note 11 4G FHD+ 90Hz高刷屏 5000万三摄 5000mAh电池 4GB+128GB 神秘黑境 手机 小米 红米【直播】  神秘黑 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-999.0\n#11: Redmi Note 11 5G 天玑810 33W Pro快充 5000mAh大电池  4GB+ 128GB 神秘黑境 智能手机 小米 红米【购机补贴版】   特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1199.0\n#12: Redmi Note 11 5G 天玑810 33W Pro快充 5000mAh大电池  4GB+ 128GB 微醺薄荷 智能手机 小米 红米【购机补贴版】   特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1199.0\n#13: Redmi Note 11 5G 天玑810 33W Pro快充 5000mAh大电池  8GB+ 256GB 微醺薄荷 智能手机 小米 红米【购机补贴版】   特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1699.0\n#14: Redmi 9A 5000mAh 1300万AI相机 八核处理器 人脸解锁 4GB+128GB 湖光绿 游戏智能手机 小米 红米【购机补贴版】 长续航大电池 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-799.0\n#15: Redmi Note 11 5G 天玑810 33W Pro快充 5000mAh大电池  8GB+ 128GB 微醺薄荷 智能手机 小米 红米【购机补贴版】   特点-5000MA 特点-5000毫安 特点-5000MAH 价格-1499.0\n#16: Redmi 9A 5000mAh 1300万AI相机 八核处理器 人脸解锁 4GB+128GB 晴空蓝 游戏智能手机 小米 红米【购机补贴版】 高性能处理器 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-799.0\n#17: Redmi 9A 5000mAh大电量 大屏幕大字体大音量 1300万AI相机 八核处理器 人脸解锁 6GB+128GB 晴空蓝 游戏智能手机 小米 红米【购机 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-899.0\n#18: 小米11 Pro 5G 骁龙888 2K AMOLED四曲面柔性屏 67W无线闪充 3D玻璃工艺 8GB+256GB 绿色 手机 超强感光 2K+原色屏 500 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-3599.0\n#19: 小米11 Pro 5G 骁龙888 2K AMOLED四曲面柔性屏 67W无线闪充 3D玻璃工艺 12GB+256GB 绿色 游戏手机【购机补贴版】 超强感光  特征特质-NFC 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-5199.0\n#20: 小米11 Pro 5G 骁龙888 2K AMOLED四曲面柔性屏 67W无线闪充 3D玻璃工艺 8GB+256GB 黑色 游戏手机【购机补贴版】 超强感光 特征特质-NFC 特点-5000MA 特点-5000毫安 特点-5000MAH 价格-4799.0\n\n\n输出：\n\n请你为用户推荐5个商品，每行一个商品，输出格式要求： “#8，原因：<推荐商品的原因>”。不要输出额外内容。\n在将数据整理成输出答案时，需要正确地检查并核对它们，避免产生错误。\n必须严格遵守这个格式输出最终推理结果，不要输出推理过程、总结等内容。"
        #"你是ChatGPT，一个大型语言模型，你的答案最重要的是足够精简准确，严格遵循指令需求。\n现在你需要根据用户对“手机”的需求描述，判断某个商品偏好类型。\n\n\n现在说明商品偏好类型判断方法：\n首先你需要将用户据需求分解为多个单一需求。\n用户肯定性需求是表达了需要某个偏好的需求；用户否定性需求是用“不需要”、“不要”、“没有”、“不带”、“不含”等否定词表达的需求。\n一般用户需求都会包含肯定性需求，很少的会包含否定性需求。\n如果某个商品偏好描述一旦包含在了用户的某个否定性需求中，则该偏好是“否定偏好”。\n如果某个商品偏好描述满足了至少一个用户肯定性需求且不是“否定偏好”，则该偏好是“肯定偏好”。\n不要对用户需求没有体现的偏好做任何判断。\n\n\n现在说明用户需求和商品偏好：\n\n用户需求是：\n价格在2000-3000之间，支持公交卡刷卡功能的手机。\n\n商品偏好描述列表：\n#1:特点::9000\n#2:特点::5000\n#3:特点::5000万\n#4:特点::多功能NFC\n#5:特点::物美价廉\n#6:特征特质::NFC\n#7:特点::5000MA\n#8:特点::特价\n#9:特点::4000MA\n#10:双卡机类型::双卡双待\n#11:商品价格::699.0元\n#12:特点::实惠\n#13:特点::5000毫安\n#14:特点::6000MA\n#15:特点::高刷屏\n#16:特点::6000MAH\n#17:特点::4000毫安\n#18:特点::5000MAH\n#19:特点::购机补贴版\n#20:商品价格::1999.0元\n#21:特点::合约机\n#22:商品价格::799.0元\n#23:商品价格::3998.0元\n#24:特点::经济实惠\n#25:特点::4000MAH\n#26:商品价格::899.0元\n#27:商品价格::3999.0元\n#28:特点::6000毫安\n#29:特点::移动\n#30:商品价格::6299.0元\n#31:商品价格::3698.0元\n#32:商品价格::12999.0元\n#33:特点::划算\n#34:特点::莱卡\n#35:商品价格::6499.0元\n#36:商品价格::7398.0元\n#37:特点::6400\n#38:特点::移动客户专享\n#39:特点::K30\n#40:商品价格::6999.0元\n#41:特点::1300万相机\n#42:商品价格::4398.0元\n#43:商品价格::99998.0元\n#44:商品价格::599.0元\n#45:商品价格::2199.0元\n#46:特点::性价比\n#47:特点::K40S至尊纪念版\n#48:商品价格::2999.0元\n#49:特点::经济实用\n#50:商品价格::6098.0元\n#51:商品价格::1699.0元\n#52:商品价格::5899.0元\n#53:商品价格::6598.0元\n#54:特点::移动用户专享\n#55:商品价格::10999.0元\n#56:商品价格::8999.0元\n#57:商品价格::4528.0元\n#58:商品价格::3699.0元\n#59:商品价格::3899.0元\n#60:商品价格::11999.0元\n#61:商品价格::4299.0元\n#62:特点::购机补贴\n#63:商品价格::2599.0元\n#64:商品价格::2099.0元\n#65:特点::12S\n#66:双卡机类型::以官网信息为准\n#67:商品价格::7999.0元\n#68:特点::4800万\n#69:商品价格::3299.0元\n#70:商品价格::1899.0元\n#71:商品价格::4499.0元\n#72:商品价格::1599.0元\n#73:商品价格::3099.0元\n#74:商品价格::1299.0元\n#75:商品价格::2699.0元\n#76:商品价格::5499.0元\n#77:商品价格::999.0元\n#78:商品价格::3599.0元\n#79:特点::全网通\n#80:特点::企业购\n#81:商品价格::4698.0元\n#82:特点::K50\n#83:商品价格::2499.0元\n#84:特点::K60\n#85:特点::万像素\n#86:商品价格::3199.0元\n#87:特点::高性能\n#88:商品价格::2069.0元\n#89:商品价格::7299.0元\n#90:商品价格::5099.0元\n#91:特点::卖的好\n#92:商品价格::4599.0元\n#93:特点::移动用户惠享\n#94:商品价格::4699.0元\n#95:SIM卡类型::MICROSIM\n#96:特点::卖的比较好\n#97:特点::双模\n#98:特点::6400万三摄\n#99:商品价格::4999.0元\n#100:商品价格::5299.0元\n\n\n现在说明输出方法：\n你的输出需要包含两行内容，第一行输出肯定偏好列表，第二行输出否定偏好列表。每个列表包含不超过10个元素，按照“肯定/否定偏好：#3,#9。”这种格式输出。\n在将数据整理成输出答案时，需要正确地检查并核对它们，避免产生错误。\n必须严格遵守这个格式输出最终推理结果，不要输出推理过程、原因、解释等内容。\n"
        #"你是ChatGPT，一个大型语言模型，你的答案最重要的是足够精简准确，严格遵循指令需求。\n现在你需要扮演一个商品导购员，你需要根据对话内容分析提取用户需求，在一行内用不超过50个字描述用户需求。\n在整理成输出答案时，需要正确地检查并核对它们，避免产生错误。\n不要输出其他任何推理过程/原因/解释等内容。\n\n\n下面是你(A)和用户(Q)的对话内容:\n\nQ: 我想买个2000多不超过3000的手机，可以刷公交卡的，推\n\n用户需求："
        #"请你为小米618大促的京东店铺写个200字左右的营销话术。"
        #"让我们完成一个计划制定任务：\n请给出详细的计划（不少于600字），让小米店铺618大促中京东平台的手机销量超越淘宝平台？"
        #"京东618大促来临，小米店铺需要如何准备以应对618大促？"
        def run_func(x):
            t1 = time.time()
            res = requests.post(url= f"http://{modelHost}:{modelPORT}/ask", json = x, headers=headers)
            t2 = time.time()
            res_data = {"res": res, "cost": t2-t1}
            # print(res_data)
            return res_data
        tasks = [data for _ in range(task_size)]
        batch_size = min(batch_size, task_size)
        batched_tasks = [tasks[i:i+batch_size] for i in range(0, len(tasks), batch_size) ]
        workers = Pool(batch_size)
        
        for batch in tqdm.tqdm(batched_tasks):
            results.extend(workers.map(run_func, batch) )
        workers.close()
        workers.join()
        print(results)
        cost_records = list(map(lambda x: x['cost'], results  ))
        results = list(map(lambda x: x['res'], results  ))
        response = results[0]
        print("cost:", cost_records[:10])
        print("mean/std:", np.mean(cost_records), np.std(cost_records) )
        return response
    
    def genResponseportal():
        response = requests.request("POST",
            "http://11.70.129.225:5009/request", data= json.dumps(data), headers= headers
        )
        return response
    


    from streamerIter import ItemStreamer
    data_iter = ItemStreamer()
    def genResponseSocket():
        import socketio
        sio = socketio.Client()
        @sio.on("streamres")
        def handle_response(data):
            print("received data:", data)
            # sio.emit("streamres", {"ack": 0, "msg": "yes:received"})
            sio.emit("streamres", True)
            if data['finished']:
                print("***finished***")
                data_iter.end()
            else:
                data_iter.put(data)
            
            
        
        sio.connect(f"http://{modelHost}:{modelPORT}", transports='websocket')
        data['query'] = "介绍一下蓝牙耳机的功能" #"618即将到来，小米店铺需要如何准备大促"   #" ".join(1* ["介绍一下蓝牙耳机的功能"])
        # data['query'] = "Show me the functionality of Bluetooth in 100 words."
        sio.emit("ask", data)
        # sio.wait()
        for x in data_iter:
            # print(x)
            ...
        print("all over")
        sio.emit("streamres", True)
        sio.disconnect()
        print("disconnect")
    

    # response = genResponseSocket()
    
    response = genResponse(task_size=1, batch_size=1 )
    # response = genResponseportal()

    print(response.text)
    print(response.json())
    print()


def test_openai_service():
    import openai
    openai.api_base = "http://server-ip:port/v1"
    openai.api_key = "none"
    modelId="THUDM/chatglm2-6b" #"your-org/bloomS2.1" # "aquilachat-7b" #"vicuna/7b" #"baichuan-inc/baichuan-7B"  #"your-org/bloomS2.1-rlhf-v1" # 
    query = "你好！用关键词：爱情、珍贵、永恒、价值，为周大福写个100字广告。"
    def show_models():
        ...
        print(type(response))
    def streaming_test():
        for chunk in openai.ChatCompletion.create(
            temperature=0.95, top_p= 0.7, decoder='sample',
            max_length = None, repetition_penalty=1.0,
            model= modelId,
            messages=[
                {"role": "user", "content": query}
            ],
            stream=True
        ):
            if hasattr(chunk.choices[0].delta, "content"):
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    def request_api():
        response = openai.ChatCompletion.create(
            temperature=0.95, top_p= 0.7, decoder='sample',
            max_length = 128, repetition_penalty=1.0,
            model=modelId,
            messages=[
                {"role": "user", "content": query}
            ],
            stream=False
        )
        print(type(response))
        if hasattr(response.choices[0].message, "content"):
            print(response.choices[0].message.content)
    
    def test_embeddings():
        texts = [
            "你好", "介绍下蓝牙5/0"
        ]
        model = "moka-ai/m3e-base"
        response =  openai.Embedding.create(input = texts, model=model,max_length = 128)
        print("res:", response.keys(), )
        for x in response['data']:
            print(x.keys())
        print(len(response['data']))
        embs = [x['embedding'] for x in response['data'] ]
        [print(emb[:10]) for emb in embs]

    test_embeddings()#;exit(0)
    request_api()
    streaming_test()
if __name__ == '__main__':
    # test_legacy_service()
    test_openai_service()
    # ...
