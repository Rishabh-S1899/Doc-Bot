import yake

def extract_keywords(text, num_keywords=3):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    temp = ' '.join(word for word, _ in keywords[:num_keywords])
    main_words = temp.split()[:num_keywords]
    answer = ' '.join(main_words)
    return answer.title()

# prompt = "Give a detailed explaination of the the breaking system of Yahama bikes"
# keywords = extract_keywords(prompt)
# print(keywords)