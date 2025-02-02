# Given list of keywords and dict containing related words of keywords. Form a paragraph with generative AI and NLP such that now word appears more than 2 times. Write pseudo code in python.

def generate_paragraph(keywords, related_words_dict):
    paragraph = []
    word_count = {}  

    def add_word(word):
        """ Add a word to the paragraph ensuring max frequency is 2 """
        if word_count.get(word, 0) < 2:
            paragraph.append(word)
            word_count[word] = word_count.get(word, 0) + 1

    for keyword in keywords:
        add_word(keyword)  
        
        if keyword in related_words_dict:
            for word in related_words_dict[keyword]:
                add_word(word)  
    return " ".join(paragraph) + "."

keywords = ["AI", "NLP", "Python"]
related_words_dict = {
    "AI": ["machine learning", "deep learning", "automation"],
    "NLP": ["text processing", "language model", "semantics"],
    "Python": ["programming", "scripting", "automation"]
}

paragraph = generate_paragraph(keywords, related_words_dict)
print(paragraph)