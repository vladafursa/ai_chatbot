#  Initialise AIML agent
import aiml
import wikipedia
from lingua import Language, LanguageDetectorBuilder
#import methods from files
from taskA import getAnswerToSimilarQuestion, getVet, translate
from taskB import getKB, checkForContradictions, appendKB, checkFact
from taskC import getImagePath,  classify
from fuzzy_logic import getDomestication, getInteger

# Create a Kernel object. 
kern = aiml.Kernel()
kern.setTextEncoding(None)
kern.bootstrap(learnFiles="mybot-basic.xml")

#assigning variables
kb = getKB()
checkForContradictions(kb)
#creating detector
detector = LanguageDetectorBuilder.from_all_languages().build()

# Welcome user
print("Welcome to cat chat bot. Please feel free to ask questions from me!")


# Main loop
while True:
    #get user input
    try:
        userInput = input("> ")
        detected_language = detector.detect_language_of((userInput)) #determining the language user used for input
        languageUsed = str(detected_language.iso_code_639_1.name.lower())
        userInput = translate(userInput, languageUsed, 'en')
        #print(languageUsed)
    except (KeyboardInterrupt, EOFError):
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
    if responseAgent == 'aiml':
        answer = kern.respond(userInput)
    #post-process the answer for commands
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        if cmd == 0:
            print(translate(params[1], 'en', languageUsed))
            break
        elif cmd == 1:
            try:
                wSummary = wikipedia.summary(params[1], sentences=3,auto_suggest=True)
                print(translate(wSummary, 'en', languageUsed))
            except:
                print(translate("Sorry, I do not know that. Be more specific!", 'en', languageUsed))
        elif cmd == 2:
                getVet(params[1])

        elif cmd == 31:# if input pattern is "I know that * is *"
            object = params[1].lower()
            subject = params[2].lower()
            appendKB(object, subject, kb)

        elif cmd == 32: # if the input pattern is "check that * is *"
            object = params[1].lower()
            subject = params[2].lower()
            checkFact(object, subject, kb)
        elif cmd == 33:
            size = getInteger("rate the size out of 10: ")
            socialBehaviour = getInteger("rate the friendliness out of 10: ")
            fit = getInteger("rate the fit out of 10: ")
            fur = getInteger("rate how soft the fur is out of 10: ")
            score = getDomestication(size, socialBehaviour, fit, fur)
            if score < 50:
                print(f"Your cat is {(100-score):.1f}% wild.")
            else:
                print(f"Your cat is {score:.1f}% domestic.")
        elif cmd == 4:
            imagePath = getImagePath()
            predictedClass = classify(imagePath)
            print("It is ", predictedClass)
        elif cmd == 99:
            print(translate(getAnswerToSimilarQuestion(userInput), 'en', languageUsed))
    else:
        print(answer)