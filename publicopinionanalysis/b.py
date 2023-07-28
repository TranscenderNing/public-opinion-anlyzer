import re

text = '1. Sentiment: Positive\nReasoning process: The content describes the achievement of the China Mobile Zhejiang Company as the "Excellent Operator of Mobile Network Quality in Key Regions," which indicates a positive sentiment towards the company.\n2. Province: Zhejiang\nReasoning process: The content specifically mentions the China Mobile Zhejiang Company as the winner of the mobile network quality evaluation rankings in key regions.\n3. Keywords: mobile network, quality, evaluation, operator, Zhejiang\nReasoning process: The main subject of the public opinion is the quality evaluation of mobile networks, and the key operators involved are China Mobile and Zhejiang Company.\n4. Business details: N/A\nReasoning process: There is no relevant information about business details in the content.\n5. Impact on users: N/A\nReasoning process: The content does not provide any information about the potential effects or consequences on users.\n6. Classification: Major public opinion\nReasoning process: The content refers to the mobile network quality evaluation rankings of key regions in China, which is a significant and important issue for the public.\n7. Manual follow-up and resolution: N/A\nReasoning process: There is no indication in the content that manual follow-up or resolution is necessary.\n'
 
matches = ['1.','2.','3.','4.','5.','6.','7.']
matchIdxes = []

for index,match in enumerate(matches):
    matchIdxes.append(text.find(match))

for i in range(len(matches)):
    if i == len(matches) - 1:
        content = text[matchIdxes[i]+2:len(text)]
    else:
        content = text[matchIdxes[i]+2:matchIdxes[i+1]]
    print(content)
    print()
    

