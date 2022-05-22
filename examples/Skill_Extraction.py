# python interpreter: ml_skills
"""
Skill Extraction
A common task is extracting competencies from unstructured text. Sometimes this is ontology-based (finding concepts from a known ontology in text), but this is not necessarily true. Skills-ML unites these with a common interface in the SkillExtractor class. The common interface is that every SkillExtractor needs to be able to take in a collection of documents, and yield what we call CandidateSkill objects.

What Is a CandidateSkill?
A CandidateSkill is a possible occurrence of a skill/competency in context in some document. It consists of the following fields:

skill_name - The text version of the skill as it appears in the document
matched_skill_identifier - A reference to the skill in some ontology.
                        This may be empty, if no ontology was used to search for skills.
context - The text surrounding the skill in the document. The goal is for a human labeler to be able to use
            this to determine whether or not the occurrence represents a true skill.
             How much context is included is up to the algorithm.
start_index - The start index of the skill occurrence within the document string.
confidence - The confidence level the algorithm has in this candidate skill being a true occurrence of a skill.
             This may be empty, if the algorithm has now way of producing a confidence value.
document_id - A unique identifier for the source document.
document_type - The type of document (examples: Job Posting, Profile, Course Description)
source_object - The entire source document.
skill_extractor_name - The name of the skill extractor algorithm. Every SkillExtractor subclass defines a name
                    property that is used to give processes downstream context about how their output data was produced.
"""

from skills_ml.algorithms.skill_extractors import SkillEndingPatternExtractor
from skills_ml.job_postings.common_schema import JobPostingCollectionSample
from skills_ml.ontologies.onet import Onet
import pandas as pd


onet = Onet()
job_posting_generator = JobPostingCollectionSample()
# extracting skills as 'all noun phrases that end in the word skill or skills'.
# instantiate the skill extractor. This class defaults to only considering lines that
# start with a bullet, which doesn't work for this dataset. So we set this flag to False.
skill_extractor = SkillEndingPatternExtractor(only_bulleted_lines=False)
job_posting = next(iter(job_posting_generator))
for candidate_skill in skill_extractor.candidate_skills(job_posting):
    print('skill name:', candidate_skill.skill_name)
    print('context:', candidate_skill.context)
    print('')


## skill extractor: matching with ONET data.
from skills_ml.algorithms.skill_extractors import ExactMatchSkillExtractor
skill_extractor = ExactMatchSkillExtractor(onet.competency_framework)
for candidate_skill in skill_extractor.candidate_skills(job_posting):
    print('skill name:', candidate_skill.skill_name)
    print('context:', candidate_skill.context)
    print('')

## SocScopedExactMatchSkillExtractor. This does exact matching, but only for the occupation that the document is tagged
# with. This, of course, is only applicable if the document has one. And it needs a full CompetencyOntology to work.
from skills_ml.algorithms.skill_extractors import SocScopedExactMatchSkillExtractor
skill_extractor = SocScopedExactMatchSkillExtractor(onet)
for candidate_skill in skill_extractor.candidate_skills(job_posting):
    print('skill name:', candidate_skill.skill_name)
    print('context:', candidate_skill.context)
    print('')

"""
Here's a list of all the other skill extractors available:
FuzzyMatchSkillExtractor - Similar to the ExactMatchSkillExtractor,
                        but using a configurable edit distance to find skill names that are very close to the targets.
AbilityEndingPatternExtractor - Similar to the SkillEndingPatternExtractor,
                                but finding noun phrases that end in 'ability' or 'abilities'.
SectionExtractSkillExtractor - Attempts to divide the text into sections with headers,
                            which is a common pattern found in job postings. Return each individual sentence found in
                             sections with certain headers (Skills, Competencies, Qualifications).
"""

from skills_ml.algorithms.skill_extractors import FuzzyMatchSkillExtractor
skill_extractor = FuzzyMatchSkillExtractor(onet.competency_framework)
# for job in job_posting_generator:
#     for candidate_skill in skill_extractor.candidate_skills(job):
#         print('job title:', candidate_skill.source_object['title'])
#         print('ONET skill:', candidate_skill.matched_skill_identifier)
#         print('skill name:', candidate_skill.skill_name)
#         print('context:', candidate_skill.context)
#         print('')


skill_match = pd.DataFrame(columns=['job title', 'ONET skill', 'skill name', 'context'])
i=0
for job in job_posting_generator:
    for candidate_skill in skill_extractor.candidate_skills(job):
        skill_match.loc[i, 'job title'] =  candidate_skill.source_object['title']
        skill_match.loc[i, 'ONET skill'] = candidate_skill.matched_skill_identifier
        skill_match.loc[i, 'skill name'] = candidate_skill.skill_name
        skill_match.loc[i, 'context'] = candidate_skill.context
        i+=1
# skill_match.to_csv("FuzzyMatchSkillExtractor.csv")


analysis = {skill.name: skill for i, skill in enumerate (onet.competencies) if ("Analysis" in skill.name) or ("analysis" in skill.name)}
analysis_begin = {skill.name: skill for i, skill in enumerate (onet.competencies) if (skill.name.startswith("Analy")) or (skill.name.startswith("Analy"))}

