import spacy

nlp = spacy.load("nl")


cat_id = nlp.vocab.strings["मैं"]
cat_vector = nlp.vocab.vectors[cat_id]
print(cat_vector == nlp.vocab["मैं"].vector)

print(cat_id in nlp.vocab)

print(len(nlp.vocab))

print(nlp.vocab["hello"].vector)

print(nlp.vocab["मैं"].vector)