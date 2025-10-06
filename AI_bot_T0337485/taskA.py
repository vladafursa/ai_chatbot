import csv
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import requests
from requests.structures import CaseInsensitiveDict

#main task
QA_FILENAME = "qa.csv"
nlp = spacy.load('en_core_web_sm')
stopWords = set(stopwords.words("english"))

#reading questions and answers from csv file into a dictionary
def readCSVFile():
    pairs = {}
    try:
        with open(QA_FILENAME, mode='r', newline='') as file:
            reader = csv.reader(file)
            for line in reader:
                question = line[0]
                answer = line[1]
                pairs[question] = answer
    except FileNotFoundError:
        print("The file doesn't exist")
    except Exception as e:
        print(f"An error opening the file occurred: {e}")
    file.close()
    return pairs

def lemmatizeSentence(sentence):
    sentence = sentence.lower() #converting the sentence to lower case
    doc = nlp(sentence) #convert the sentence to tokens
    lemmatizedSentence = " ".join([token.lemma_ for token in doc if token.text not in stopWords]) #building the sentence with lemmas and checking that it is not a stopword
    return lemmatizedSentence

def getAnswerToSimilarQuestion(userInput):
    pairs = readCSVFile()  # reading questions and answers
    questionsFromCSV = []
    questionsFromCSV = list(pairs.keys()) #appending with questions
    questionsFromCSV.append(userInput) #adding user's question
    preprocessed_docs = [lemmatizeSentence(doc) for doc in questionsFromCSV] #lemmatising questions
    vectorizer = TfidfVectorizer()
    tfidfMatrix = vectorizer.fit_transform(preprocessed_docs) #vectorization
    cosineSim = cosine_similarity(tfidfMatrix[-1], tfidfMatrix[:-1]) #cosine similary between the last sentence and other elements of a matrix

    bestMatchIndex = cosineSim.argmax() #finding index of the question with the highest similarity
    bestMatchScore = cosineSim[0, bestMatchIndex] #getting matching score of it
    if bestMatchScore > 0: #check for a minimal threshold
        bestMatchedQuestion = questionsFromCSV[bestMatchIndex] #retrieving question with the highest similarity
        bestMatchedAnswer = pairs[bestMatchedQuestion] #retrieving answer on it
        return bestMatchedAnswer
    else:
        return "I didn't get the question, please repeat"


#additional of getting the closes vet service to the provided postcode
def convertPostcode(postcode):
    url = "http://api.getthedata.com/postcode/" + postcode
    resp = requests.get(url)
    jsonData = resp.json()
    return jsonData
def getVet(postcode):
    response_data = convertPostcode(postcode)
    lat = response_data['data']['latitude']
    lon = response_data['data']['longitude']
    API_KEY = "b197376e709248b2a056f2ee02d9230d"
    url = "https://api.geoapify.com/v2/places?categories=pet.veterinary&filter=circle:" + lon + ","+ lat +",5000&bias=proximity:" + lon + ","+ lat +"&limit=1&apiKey=" + API_KEY
    headers = CaseInsensitiveDict()
    headers["Accept"] = "application/json"
    resp = requests.get(url, headers=headers)
    jsonData = resp.json()
    print(jsonData['features'][0]['properties']['formatted'])


#extra task: translation
def translate(textToTranslate, inputLanguage, targetLanguage):
    if inputLanguage != targetLanguage: #to not translate sentence to itself
        translator = GoogleTranslator(source=inputLanguage, target=targetLanguage) #initialising translator: from what to what language to translate
        translatedText = translator.translate(textToTranslate)
        return translatedText
    else:
        return textToTranslate