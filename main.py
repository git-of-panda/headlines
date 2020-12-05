from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
from spacy.lang.en import English

def format_text(text):
    article = text.split(".")
    sentences = []

    en = English()

    art = en(text)
    for al in list(art.sents):
        print("\"%s\"" %al.string)
    #for sentence in article:
    #    sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    #    print("Sentence: ", sentence)
    #sentences.pop() 
    
    #return sentences

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    # build the vector for the first sentence
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
    # build the vector for the second sentence
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)
 
def build_similarity_matrix(sentences, stop_words):
    # Create an empty similarity matrix
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: #ignore if both are same sentences
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix


def generate_summary(text, top_n):
    stop_words = stopwords.words('english')
    summarize_text = []

    # Step 1 - Read text and split it
    sentences =  format_text(text)

    # Step 2 - Generate Similary Martix across sentences
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

    # Step 3 - Rank sentences in similarity martix
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

    # Step 4 - Sort the rank and pick top sentences
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    for i in range(top_n):
      summarize_text.append(" ".join(ranked_sentence[i][1]))

    # Step 5 - Offcourse, output the summarize texr
    print("Summarize Text: ", ".\n\nSAD ".join(summarize_text))

if __name__ == "__main__":
    text_to_summarise = """Man named Adolf Hitler Uunona wins election in Namibia, says he isn’t seeking ‘world domination’.
    A man named Adolf Hitler Uunona has won a local election in Namibia with 85 percent of the vote, becoming a councilor in the Ompundja Constituency.
According to the Electoral Commission of Namibia, Uunona claimed a seat for the South West Africa People’s Organization, also known as the SWAPO party, after racking up 1,196 votes last month.
Uunona has no connection to his unusual namesake — but does have a sense of humor about the situation. He is not “striving for world domination” and does not have plans to “conquer” the Oshana region, he told German newspaper Bild.
On the official candidates list, the name Hitler was reduced to an initial, with the document reading: Adolf H. Uunona, although the results page listed Uunona’s name in full.
After Uunona’s landslide victory, the term “Adolf Hitler” became a top Twitter trend Thursday, as people around the world discussed his name.
“Of course, 2020 would not be complete if Adolf Hitler didn’t win an election with 85% of the vote,” one Twitter user wrote, and others asked: “Is that his real name?”
In his interview with Bild, Uunona said that his father had named him after the Nazi dictator but “probably didn’t understand what Adolf Hitler stood for.” His wife, he said, calls him Adolf.
Ahead of the November regional vote, Electoral Commission Chairwoman Notemba Tjipueja urged people to exercise their democratic right and cast their ballots “in large numbers.” Uunona’s political opponent Mumbala Abner of the Independent Patriots for Change party received 213 votes.
The southwestern African country is a former German colony, and a number of streets and places still have German names.
In August, the country rejected a German offer of compensation for the slaughter of tens of thousands of Herero and Nama peoples at the hands of German colonial forces between 1904 and 1908. An estimated 80,000 people are believed to have been killed by troops, which Germany has since called a “terrible crime.”
Namibian President Hage Geingob said at the time that the offer was “not acceptable” and would need to be “revised,” the Guardian reported.
Uunona is not the first politician overseas to garner global interest because of an unusual name.
In November, the Japanese mayor of a small town in the country’s Kumamoto prefecture rose to fame after the result of the U.S. presidential election.
Under the Japanese writing system, Yutaka Umeda’s name can also be pronounced “Jo Baiden.”
Umeda told local media outlets that he was oblivious to the connection to the U.S. Biden, who is set to assume the presidency in January, until his family members alerted him that people on social media were talking about him.
“It feels as though I’ve also won the election,” Umeda said in response to his newfound fame.
In 2012, the Indian state of Gujarat unveiled a men’s clothing store named “Hitler,” with a swastika on display inside the dot of the “i” on the shop’s sign. The choice of name sparked fierce backlash from Jewish groups.
Abraham H. Foxman, a Holocaust survivor and director of the Anti-Defamation League, said that the move was a “perverse abuse of the history of the Holocaust,” adding that it was inexcusable to name a business “after one of the world’s most notorious mass murders and anti-Semites.”
"""
    generate_summary( text_to_summarise, 5)