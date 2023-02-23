# Import necessary libraries
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv("new_train.csv")

# Preprocess the data
vectorizer = CountVectorizer(stop_words="english", max_df=0.2, min_df=21, ngram_range=(1, 2))
X = vectorizer.fit_transform(df["transcription"])
y = df["medical_specialty"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Train the model
clf = MultinomialNB(alpha=0.01)
clf.fit(X_train, y_train)

# Test the model
y_pred = clf.predict(X_test)

# Evaluate the model
#print(classification_report(y_test, y_pred))

from sklearn.metrics import f1_score
macro_f1 = f1_score(y_test, y_pred, average='macro')
micro_f1 = f1_score(y_test, y_pred, average='micro')
print("Macro F1-score: {:.9f}".format(macro_f1))
print("Micro F1-score: {:.9f}".format(micro_f1))


# Predict the specialty of a new transcription and find confidence of prediction
new_transcription = "HISTORY: , A 34-year-old male presents today self-referred at the recommendation of Emergency Room physicians and his nephrologist to pursue further allergy evaluation and treatment.  Please refer to chart for history and physical, as well as the medical records regarding his allergic reaction treatment at ABC Medical Center for further details and studies.  In summary, the patient had an acute event of perioral swelling, etiology uncertain, occurring on 05/03/2008 requiring transfer from ABC Medical Center to XYZ Medical Center due to a history of renal failure requiring dialysis and he was admitted and treated and felt that his allergy reaction was to Keflex, which was being used to treat a skin cellulitis dialysis shunt infection.  In summary, the patient states he has some problems with tolerating grass allergies, environmental and inhalant allergies occasionally, but has never had anaphylactic or angioedema reactions.  He currently is not taking any medication for allergies.  He is taking atenolol for blood pressure control.  No further problems have been noted upon his discharge and treatment, which included corticosteroid therapy and antihistamine therapy and monitoring.,PAST MEDICAL HISTORY:,  History of urticaria, history of renal failure with hypertension possible source of renal failure, history of dialysis times 2 years and a history of hypertension.,PAST SURGICAL HISTORY:,  PermCath insertion times 3 and peritoneal dialysis.,FAMILY HISTORY: , Strong for heart disease, carcinoma, and a history of food allergies, and there is also a history of hypertension.,CURRENT MEDICATIONS: , Atenolol, sodium bicarbonate, Lovaza, and Dialyvite.,ALLERGIES: , Heparin causing thrombocytopenia.,SOCIAL HISTORY: , Denies tobacco or alcohol use.,PHYSICAL EXAMINATION:  ,VITAL SIGNS:  Age 34, blood pressure 128/78, pulse 70, temperature is 97.8, weight is 207 pounds, and height is 5 feet 7 inches.,GENERAL:  The patient is healthy appearing; alert and oriented to person, place and time; responds appropriately; in no acute distress.,HEAD:  Normocephalic.  No masses or lesions noted.,FACE:  No facial tenderness or asymmetry noted.,EYES:  Pupils are equal, round and reactive to light and accommodation bilaterally.  Extraocular movements are intact bilaterally.,EARS:  The tympanic membranes are intact bilaterally with a good light reflex.  The external auditory canals are clear with no lesions or masses noted.  Weber and Rinne tests are within normal limits.,NOSE:  The nasal cavities are patent bilaterally.  The nasal septum is midline.  There are no nasal discharges.  No masses or lesions noted.,THROAT:  The oral mucosa appears healthy.  Dental hygiene is maintained well.  No oropharyngeal masses or lesions noted.  No postnasal drip noted.,NECK:  The neck is supple with no adenopathy or masses palpated.  The trachea is midline.  The thyroid gland is of normal size with no nodules.,NEUROLOGIC:  Facial nerve is intact bilaterally.  The remaining cranial nerves are intact without focal deficit.,LUNGS:  Clear to auscultation bilaterally.  No wheeze noted.,HEART:  Regular rate and rhythm.  No murmur noted.,IMPRESSION:  ,1.  Acute allergic reaction, etiology uncertain, however, suspicious for Keflex.,2.  Renal failure requiring dialysis.,3.  Hypertension.,RECOMMENDATIONS:  ,RAST allergy testing for both food and environmental allergies was performed, and we will get the results back to the patient with further recommendations to follow.  If there is any specific food or inhalant allergen that is found to be quite high on the sensitivity scale, we would probably recommend the patient to avoid the offending agent to hold off on any further reactions.  At this point, I would recommend the patient stopping any further use of cephalosporin antibiotics, which may be the cause of his allergic reaction, and I would consider this an allergy.  Being on atenolol, the patient has a more difficult time treating acute anaphylaxis, but I do think this is medically necessary at this time and hopefully we can find specific causes for his allergic reactions.  An EpiPen was also prescribed in the event of acute angioedema or allergic reaction or sensation of impending allergic reaction and he is aware he needs to proceed directly to the emergency room for further evaluation and treatment recommendations after administration of an EpiPen."
new_transcription_vec = vectorizer.transform([new_transcription])
prediction = clf.predict(new_transcription_vec)
predicted_prob = clf.predict_proba(new_transcription_vec)

# Print the predicted specialty
print("Predicted specialty:", prediction)
print()
