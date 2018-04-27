# Extracting skills using noun phrase endings
#
# To showcase the noun phrase skill extractor, we run a sample of job postings through it.
# In the end, we have the most commonly occurring noun phrases ending in
# 'skill' or 'skills'.
from collections import Counter
import logging

from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.job_postings.corpora.basic import SimpleCorpusCreator
from skills_ml.algorithms.skill_extractors.noun_phrase_ending import SkillEndingPatternExtractor

logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    # Use the simplest possible input:
    # 1. 50 pre-downloaded job postings
    job_postings = JobPostingCollectionSample()

    # 2. A standard transformation into a plaintext corpus, pulling a few text-heavy fields
    corpus = SimpleCorpusCreator(JobPostingCollectionSample())

    # 3. A skill extractor to retrieve noun phrases ending in 'skill' or 'skills'.
    # VT job postings do not include line breaks, so the bulleted-line filter
    # will remove all possible matches. Let's turn it off
    pattern_extractor = SkillEndingPatternExtractor(only_bulleted_lines=False)

    skill_counts = Counter()
    for document in corpus:
        skill_counts += pattern_extractor.document_skill_counts(document)

    logging.info('10 Most Common Skills in job descriptions: %s', skill_counts.most_common(10))
