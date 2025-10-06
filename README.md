The system can answer user’s questions about cats: both wild and domestic. The chatbot can conduct similarity-based conversation. If the command provided by user is not found, similar question is searched in qa.csv file using lemmatisation, tf/idf and cosine similarity techniques. If the question is found, answer for it is outputted. The chatbot can recognize languages and respond in the language used. For translation DeepL library was used, for language detection - lingua. The chatbot can provide the closest vet to specified postcode by simple API calls.

Knowledge base with FOL statements was created as a logical-kb.csv. Before the start of the program, it is checked for contradictions, and if they are present, the program is terminated. The user can expand in-memory knowledgebase by typing “I know that * is *”. User’s statement is checked to not contradict with the existing knowledgebase. If there is no contradiction, the statement is added. If not, “Contradiction” message is shown. The user can also check if his statement is correct, incorrect or the model doesn’t know about it by typing “Check that * is *”. All mentioned is done with nltk library. The user can ask for the closest vet to the provided postcode. It is done with API query.

The user can also determine to what extent the cat is wild or domestic by inputting rates (0 to 10) on 4 parameters: size, social behaviour, fit, softness of fur. The output will be % of being wild or domestic. The % of being domestic is calculated, if it is < 50, 100 - % is shown for being wild. This is done using fuzzy logic: skfuzzy library.

The user can distinguish between 3 species of cats: lion, tiger and cheetah with the help of image classification. The user is asked to provide file path to the image. A prediction of what is on the image is outputted. Image classification was done using tensorflow and keras libraries. For the best accuracy pretrained VGG16 was used


<img width="521" height="578" alt="Screenshot 2025-10-07 at 02 44 44" src="https://github.com/user-attachments/assets/95238b0c-8434-471e-ba8d-f5c093cf3850" />
<img width="524" height="408" alt="Screenshot 2025-10-07 at 02 45 08" src="https://github.com/user-attachments/assets/27eb8543-c14d-44e2-b023-2735bb475726" />
<img width="519" height="255" alt="Screenshot 2025-10-07 at 02 45 24" src="https://github.com/user-attachments/assets/9e96217e-62ef-4c72-8371-5d56b4e136e3" />
<img width="516" height="473" alt="Screenshot 2025-10-07 at 02 46 04" src="https://github.com/user-attachments/assets/e750a0a0-2b29-4403-aee8-73d815e4de28" />
<img width="523" height="574" alt="Screenshot 2025-10-07 at 02 46 33" src="https://github.com/user-attachments/assets/42a6f2ba-9301-4c5b-b12d-60676a1ef249" />
<img width="516" height="332" alt="Screenshot 2025-10-07 at 02 46 57" src="https://github.com/user-attachments/assets/cd2cee25-e6e8-43f8-94e9-87489ac0f66e" />
<img width="518" height="422" alt="Screenshot 2025-10-07 at 02 47 14" src="https://github.com/user-attachments/assets/577e1327-9f7e-48f8-9f07-1a8f10bed54d" />



