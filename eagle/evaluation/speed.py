# import json
# from transformers import AutoTokenizer
# import numpy as np

# tokenizer=AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
# setting_add_dict = {
# "jsonl_file_eagle" : "/home/farinneya/mt_bench/ess-vicuna-70b-fp16-temperature-1.0.jsonl",
# "jsonl_file_3xMOE_top2" : "/home/farinneya/mt_bench/custom_eagle-temperature-1.0.jsonl",
# "jsonl_file_3xMOE_top1" : "/home/farinneya/mt_bench/custom_eagle_topk1-temperature-1.0.jsonl",
# }
# # jsonl_file_baseline = "/home/farinneya/mt_bench/ess-vicuna-70b-fp16-baseline-temperature-1.0.jsonl"
# json_file_baseline = "/home/farinneya/mt_bench/ess-vicuna-70b-fp16-temperature-1.0.jsonl"

# data = []
# with open(setting_add_dict["jsonl_file_3xMOE_top2"], 'r', encoding='utf-8') as file:
#     for line in file:
#         json_obj = json.loads(line)
#         data.append(json_obj)



# speeds=[]
# for datapoint in data:
#     qid=datapoint["question_id"]
#     answer=datapoint["choices"][0]['turns']
#     tokens=sum(datapoint["choices"][0]['new_tokens'])
#     times = sum(datapoint["choices"][0]['wall_time'])
#     speeds.append(tokens/times)


# data = []
# with open(setting_add_dict["jsonl_file_eagle"], 'r', encoding='utf-8') as file:
#     for line in file:
#         json_obj = json.loads(line)
#         data.append(json_obj)


# total_time=0
# total_token=0
# speeds0=[]
# for datapoint in data:
#     qid=datapoint["question_id"]
#     answer=datapoint["choices"][0]['turns']
#     tokens = 0
#     for i in answer:
#         tokens += (len(tokenizer(i).input_ids) - 1)
#     times = sum(datapoint["choices"][0]['wall_time'])
#     speeds0.append(tokens / times)
#     total_time+=times
#     total_token+=tokens



# print("moe top2 eagle",np.array(speeds).mean())
# print('eagle ',np.array(speeds0).mean())
# print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())


# # {"question_id": 81, "answer_id": "FEKmwUZPWvhrgaG9TXVXQs", "model_id": "ess-vicuna-70b-fp16-baseline-temperature-1.0", "choices": [{"index": 0, "turns": ["Aloha! Recently, I had the opportunity to visit the beautiful state of Hawaii, and it was truly an unforgettable experience. From the crystal-clear waters to the rich culture and history, Hawaii has something for everyone. In this blog post, I'll be sharing some of the highlights of my trip, including some of the cultural experiences and must-see attractions that I would recommend to anyone planning a visit to this tropical paradise.\n\nOne of the first things that struck me about Hawaii was the strength and resilience of the local community. Despite facing numerous challenges, including natural disasters and economic struggles, the people of Hawaii have remained positive and welcoming. I had the opportunity to visit the Ka'unakakai Harbor on the island of Molokai, which is a fishing community that has been impacted by overfishing and climate change. Despite these challenges, the community is actively working to preserve their way of life and maintain their cultural heritage. It was truly inspiring to see the strength and determination of the people of Hawaii.\n\nAnother highlight of my trip was exploring the many cultural experiences that Hawaii has to offer. I had the opportunity to visit the Polynesian Cultural Center, which is a must-see attraction for anyone visiting Hawaii. The center showcases the cultures of Polynesia, including traditional dances, music, and crafts. I also had the opportunity to learn about the history and traditions of the indigenous people of Hawaii, including the Hawaiian language, music, and dance. It was a truly immersive experience that gave me a glimpse into the rich culture and heritage of the islands.\n\nOf course, one of the main draws of Hawaii is the natural beauty of the islands. I had the opportunity to visit several national parks and preserves on the islands, including Haleakala National Park on the island of Maui and Volcanoes National Park on the island of Hawaii. Both parks offer breathtaking views of the natural beauty of the islands, including volcanoes, waterfalls, and rainforests. I also had the opportunity to participate in several outdoor activities, including hiking, swimming, and snorkeling. It was truly awe-inspiring to be surrounded by such natural beauty.\n\nIn conclusion, my trip to Hawaii was an unforgettable experience that I would highly recommend to anyone planning a visit to the islands. From the rich culture and heritage to the stunning natural beauty, Hawaii has something for everyone. I had the opportunity to experience many of the highlights of the islands, including the Polynesian Cultural Center and several national parks, and I'm already planning my next trip back. Mahalo for reading!", "Aloha! Recently, I had the opportunity to visit the beautiful state of Hawaii, and it was truly an unforgettable experience. From the crystal-clear waters to the rich culture and history, Hawaii has something for everyone. In this blog post, I'll be sharing some of the highlights of my trip, including some of the cultural experiences and must-see attractions that I would recommend to anyone planning a visit to this tropical paradise.\n\nOne of the first things that struck me about Hawaii was the strength and resilience of the local community. Despite facing numerous challenges, including natural disasters and economic struggles, the people of Hawaii have remained positive and welcoming. I had the opportunity to visit the Ka'unakakai Harbor on the island of Molokai, which is a fishing community that has been impacted by overfishing and climate change. Despite these challenges, the community is actively working to preserve their way of life and maintain their cultural heritage. It was truly inspiring to see the strength and determination of the people of Hawaii.\n\nAnother highlight of my trip was exploring the many cultural experiences that Hawaii has to offer. I had the opportunity to visit the Polynesian Cultural Center, which is a must-see attraction for anyone visiting Hawaii. The center showcases the cultures of Polynesia, including traditional dances, music, and crafts. I also had the opportunity to learn about the history and traditions of the indigenous people of Hawaii, including the Hawaiian language, music, and dance. It was a truly immersive experience that gave me a glimpse into the rich culture and heritage of the islands.\n\nOf course, one of the main draws of Hawaii is the natural beauty of the islands. I had the opportunity to visit several national parks and preserves on the islands, including Haleakala National Park on the island of Maui and Volcanoes National Park on the island of Hawaii. Both parks offer breathtaking views of the natural beauty of the islands, including volcanoes, waterfalls, and rainforests. I also had the opportunity to participate in several outdoor activities, including hiking, swimming, and snorkeling. It was truly awe-inspiring to be surrounded by such natural beauty.\n\nIn conclusion, my trip to Hawaii was an unforgettable experience that I would highly recommend to anyone planning a visit to the islands. From the rich culture and heritage to the stunning natural beauty, Hawaii has something for everyone. I had the opportunity to experience many of the highlights of the islands, including the Polynesian Cultural Center and several national parks, and I'm already planning my next trip back. Mahalo for reading!"], "idxs": [603, 603], "new_tokens": [0, 0], "wall_time": [10.100712299346924, 10.05989146232605]}], "tstamp": 1736976766.836729}



import json
from transformers import AutoTokenizer
import numpy as np

tokenizer=AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.3")
jsonl_file = "/home/farinneya/eagle/mt_bench/EAGLE_VANILLA-temperature-1.0.jsonl"
jsonl_file_base = "/home/farinneya/eagle/mt_bench/EAGLE_MOE_3EXP_TOP2-temperature-1.0.jsonl"
data = []
with open(jsonl_file, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)



speeds=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens=sum(datapoint["choices"][0]['new_tokens'])
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds.append(tokens/times)


data = []
with open(jsonl_file_base, 'r', encoding='utf-8') as file:
    for line in file:
        json_obj = json.loads(line)
        data.append(json_obj)


total_time=0
total_token=0
speeds0=[]
for datapoint in data:
    qid=datapoint["question_id"]
    answer=datapoint["choices"][0]['turns']
    tokens = 0
    for i in answer:
        tokens += (len(tokenizer(i).input_ids) - 1)
    times = sum(datapoint["choices"][0]['wall_time'])
    speeds0.append(tokens / times)
    total_time+=times
    total_token+=tokens



print(f'Speed up ratio of {jsonl_file} over {jsonl_file_base}')
print("ratio",np.array(speeds).mean()/np.array(speeds0).mean())