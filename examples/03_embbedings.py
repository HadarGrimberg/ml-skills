"""
Embedding Models
Currently, Skills-ML includes word2vec, doc2vec and fastext and may include more in the future.

Word2VecModel is able to look up a word vector and infer a sentence/paragraph vector by averaging each word
            in a sentence/paragraph. It supports online learning. For out-of-vocabulary word handling of
            sentence/paragraph inference, a random vector will be assigned with the same dimension.
Doc2VecModel is able to look up a word vector and infer a sentence/paragraph vector by gradient descending on the fly,
            so it is non-deterministic. It does not support online learning.
FastTextModel is able to look up a word vector and infer a sentence/paragraph vector by averaging each word
            in a sentence/paragraph. It supports online learning. For out-of-vocabulary word handling of
            sentence/paragraph inference, it sums all vectors of the unseen word’s char-ngrams. If none of the
            char-ngrams of the unseen word is present, a random vector will be assigned with the same dimension.
"""

# check the classes in models:
import inspect
from skills_ml.algorithms.embedding import models

models_classes=[name for name, obj in inspect.getmembers(models) if inspect.isclass(obj)]

from skills_ml.algorithms.embedding.models import Word2VecModel, FastTextModel

cbow = Word2VecModel(size=200, sg=0, window=7, iter=3, batch_words=1000)
skip_gram = Word2VecModel(size=200, sg=1, window=7, iter=3, batch_words=1000)
fasttext = FastTextModel(size=200, window=7, iter=3, batch_words=1000)
"""
Corpora
Next, we need some text corpus to train embedding modelss. Skills-ML provides pre-defined classes to convert common
schema job listings into a corpus in documnet level suitable for use by machine learning algorithms or specific tasks.
"""

from skills_ml.job_postings.corpora import Word2VecGensimCorpusCreator, Doc2VecGensimCorpusCreator
from skills_ml.job_postings.sample import JobSampler
from skills_ml.job_postings.filtering import JobPostingFilterer
from skills_ml.job_postings.raw.virginia import VirginiaTransformer
from urllib.request import urlopen
import json

va_url = "http://opendata.cs.vt.edu/dataset/ab0abac3-2293-4c9d-8d80-22d450254389/resource/074f7e44-9275-4bba-874e-4795e8f6830c/download/openjobs-jobpostings.may-2016.json"
has_soc = lambda x: x['onet_soc_code']
not_unknown_soc = lambda x: x['onet_soc_code'][:2] != '99'

class VAJobposting(object):
    def __init__(self, uri):
        self.uri = uri

    def __iter__(self):
        request = urlopen(self.uri)
        for line in request.readlines():
            raw = json.loads(line)
            yield VirginiaTransformer(partner_id="va")._transform(raw)


jobpostings_va = VAJobposting(va_url)


jobpostings_filtered = JobPostingFilterer(
    job_posting_generator=VAJobposting(va_url),
    filter_funcs=[has_soc, not_unknown_soc]
)

sampler = JobSampler(job_posting_generator=jobpostings_filtered, k=5000, key=lambda x: x['onet_soc_code'][:2], weights=weights)
w2v_corpus_generator = Word2VecGensimCorpusCreator(sampler)

"""
Preprocessing
Another option is: we can build our own corpus generator by using some preprocessing tools.

Function Compostition:
    ProcessingPipeline will compose processing functions together to become a callable object that takes in the input
                    from the very first processing function and returns the output of the last processing function.
    IterablePipeline will compose processing functions together to be passed to different stages(training/ prediction)
"""
from skills_ml.algorithms.preprocessing import IterablePipeline
from skills_ml.algorithms import nlp
from functools import partial

document_schema_fields = ['description','experienceRequirements', 'qualifications', 'skills']

pipeline = IterablePipeline(
    partial(nlp.fields_join, document_schema_fields=document_schema_fields),
    nlp.clean_html,
    nlp.clean_str,
    nlp.word_tokenize,
)

corpus_generator = pipeline(sampler)
"""
Train Embedding
The EmbeddingTrainer provides online batch learning for Word2VecModel and FastTextModel."""

from skills_ml.algorithms.embedding.train import EmbeddingTrainer

trainer = EmbeddingTrainer(cbow, skip_gram, fasttext, batch_size=100)
trainer.train(corpus_generator)

"""
Storage
Skills-ML has couple useful storage classes that could benefit both local or cloud.

S3Store: S3 storage engine
FSStore: File system storage engine
ModelStorage: Serialization model storage.
"""

from skills_ml.storage import FSStore, S3Store, ModelStorage

fs = FSStore(path="tmp/model_cache/embedding/examples")
trainer.save_model(storage=fs)
print(cbow.model_name)
print(cbow.storage)