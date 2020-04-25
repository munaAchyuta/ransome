## pre work views useful for further work (specifically on word2vec with crf for named entity recognition or ner)

Wang and Manning (2013) showed that linear architectures perform better in high-dimensional discrete feature space than non-linear ones,
whereas non-linear architectures are more effec-tive in low-dimensional and continuous feature space. Hence, the previous method that directly
uses the continuous word embeddings as features in linear models (CRF) is inappropriate. Word embeddings may be better utilized in the linear modeling framework by smartly transforming the
embeddings to some relatively higher dimensional and discrete representations.

* http://ir.hit.edu.cn/~jguo/papers/emnlp2014-semiemb.pdf
* http://cogcomp.org/files/presentations/BGU_WordRepresentations_2009.pdf
* https://graphaware.com/nlp/2018/09/10/deep-text-understand-combining-graphs-ner-word2vec.html
* http://www.davidsbatista.net/blog/2018/12/06/Word_Embeddings/
* NER with less data -- https://arxiv.org/pdf/1806.04411.pdf
* FREME NER TOOL -- https://freme-project.github.io/knowledge-base/freme-for-api-users/freme-ner.html
* crf+cluster  vector+dictionary+embeddings for disease NER -- http://www.dialog-21.ru/media/3932/miftahutdinovzshetal.pdf
* ultra fine entity typing.(http://nlp.cs.washington.edu/entity_type/slides.pdf)(https://github.com/uwnlp/open_type#eusol-choi-omer-levy-yejin-choi-and-luke-zettlemoyer-acl-2018)
* Fine-Grained Entity Typing in Hyperbolic Space (https://github.com/nlpAThits/figet-hyperbolic-space)
* Fine-grained Entity Typing through Increased Discourse Contextand Adaptive Classification Thresholds (https://arxiv.org/pdf/1804.08000v1.pdf)

## Named entity resolution and linking
  * The problem is twofold: in a corpus of documents, I need to identify references to a specific reference to a lab measurement (named entity recognition #1). If there is an associated value , I would like to return that as well (NER # 2). Lastly, I want to find the specific reference to a date in the document that relates to when the measurement was taken (NER#3).
  * Those three tasks form "part 1". "Part 2" is correctly linking a lab_text to value and date.
  * The catch (maybe?) is that there could be multiple of any of these in a document. Multiple references to the lab / values and multiple dates (though only one true date for each specific measurement).

It depends on how much data you have. I can see AllenNLP's Reading Comprehension could give some answers without any training. Alternative would be training your model on attention/transformer on pretrained language model. Also doesn't require much training, BERT paper said it is an hour or two on fine tuning SQuad.

This is a very hard problem, you are not over complicating it! My company sells this capability, and it has taken a TON of work and IP to build. Still, all implementations across the industry struggle at doing this well. You actually have three "parts" here:

NER: on both the "measurements" and the values of the measurements

Information Extraction: determining which "value" expression/entity in the text relates to which measurement entity in the sentence

Named Entity Resolution / Linking: linking a non-standard expression in the text to a standardized taxonomy/canonical form like "temp was 45 C" -> "Temperature, 45, Celsius"

The first thing I would try is to do this purely rules based. You didn't mention how big the list is of different types of values and measurements. If its possible to do with rules, you should 100% do this with rules, don't use ML. If you're in python, look up spaCy's entity ruler for part 1, set up rules on the parse or dependency tree for 2, and use regex, lemmatization and a lookup table for 3.

I am warning you now, getting more sophisticated than rules-based is going to take a TON of effort, and this will only make sense if you have a big time budget and resources to get labels, as well as you have enough documents you need to process that the value you get out of this is worth the cost/ doing this extraction manually costs more than the ML approach.

The first thing you are going to need is a TON of labels. This is the least glamorous but most important part. Check out spacy's BILOU markup for how to structure the labels for NER using character offsets and BILOU tags. You'd probably still need to use rules on the parse tree for 2. For 3 you could use purely rules as mentioned above, or a classifier that takes as an input the edit distance between the NER'd expression from the text, and each entry in your dictionary. Again, this is more labels that would need to be collected.

weak supervision -- The Snorkel tutorial https://www.snorkel.org/use-cases/spouse-demo
