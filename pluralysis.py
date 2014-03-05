import nltk

# use wordnet lemmatizer to determine if a word is singular or plural
# courtesy of http://stackoverflow.com/questions/18911589/how-to-test-whether-a-word-is-in-singular-form-or-not-in-python
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def check_plural(word):
    lemma = wnl.lemmatize(word, 'n')
    plural = True if word is not lemma else False
    return plural, lemma

# load tagged brown corpus using simplified POS tags 
brown_tagged = nltk.corpus.brown.tagged_words(simplify_tags=True)

# find the frequency of nouns in the corpus
tag_fd = nltk.FreqDist(word for (word, tag) in brown_tagged if tag in ['N', 'NP', 'PRO'])

# load into a dictionary of singular and plurals
all_nouns_dict = {}
for word in tag_fd.items():
  if len(word[0]) < 2:
    # ignore nouns that are one letter (e.g. us is considered plural of u by lemmatizer so ignore u)
    continue

  # build a dictionary of words with their singular and plural counts
  is_plural, lemma = check_plural(word[0]) 
  if not all_nouns_dict.has_key(lemma):
    all_nouns_dict[lemma] = {} 
  if is_plural:
    # save plural count
    all_nouns_dict[lemma]['plural_count'] = word[1]
    all_nouns_dict[lemma]['plural'] = word[0]
  else:
    # save singular count
    all_nouns_dict[lemma]['singular_count'] = word[1]
    all_nouns_dict[lemma]['singular'] = word[0]

# find words with both singular and plural and calculate plural percentage
singular_plural_dict = {}
for key, line in all_nouns_dict.iteritems():
  if line.has_key("plural") and line.has_key("singular"):
    # one could put some thresholds on number of times a word appeared to limit the results
    plural_percent = line['plural_count']/float(line['singular_count']+line['plural_count'])
    singular_plural_dict[key] = all_nouns_dict[key]
    singular_plural_dict[key]['plural_percent'] = plural_percent

# sort the dictionary that has both singular and plurals
newly_sorted = sorted(singular_plural_dict.itervalues(), key=lambda x: x['plural_percent'], reverse=True)

# output to file
f = open("singular_plural_stat.txt", "w")
f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format("Singular", "Plural", "S Count", "P Count", "P %"))

for line in newly_sorted:
  f.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(line['singular'], line['plural'], \
                                            line['singular_count'], line['plural_count'], \
                                            line['plural_percent']))

f.close()

