# python interpreter: ml_skills

from skills_ml.ontologies.onet import Onet

"""
To move on we'll want to level up to a full ontology. The example we'll use is O*NET,
built from survey data and maintained by the US Department of Labor. 
A CompetencyOntology subclass that downloads the source files from the O*NET web site is included in Skills-ML.
"""
# CompetencyOntology
onet = Onet()
onet.print_summary_stats()

# list the competencies
list(onet.competencies)[0:5]
Underground_mining_bulldozer = list(onet.competencies)[0]
Abilities = [skill for skill in onet.competencies if "Abilities" in skill.categories]
print(onet.filter_by.__code__.co_varnames)

"""
filter_by filters using edges: the filtering function it expects takes in an edge (between a Competency and Occupation)
and returns whether or not it should be in the result. The result takes the form of a new CompetencyOntology,
so you can interact with it in the same way as you would the source ontology.
"""
# Filter all competencies of nurse practitioners
nurse_practitioners = onet.filter_by(lambda edge: 'Nurse Practitioners' in edge.occupation.name)
# nurse practitioners' competency categories
np_competency_cat = set(cat for competency in nurse_practitioners.competencies for cat in competency.categories)
# find nurse practitioners' competencies from specific category
nurse_practitioners_knowledge_competency = [competency for competency in nurse_practitioners.competencies if 'Knowledge' in competency.categories]
# find nurse practitioners' competencies names from specific category
nurse_practitioners_T2_competency = [competency.name for competency in nurse_practitioners.competencies if 'O*NET T2' in competency.categories]



## Job Posts

##URL is not working, can't handle it
from skills_ml.job_postings.raw.virginia import VirginiaTransformer
from urllib.request import urlopen
import json

va_url = "http://opendata.cs.vt.edu/dataset/ab0abac3-2293-4c9d-8d80-22d450254389/resource/074f7e44-9275-4bba-874e-4795e8f6830c/download/openjobs-jobpostings.may-2016.json"


class VAJobposting(object):
    def __init__(self, uri):
        self.uri = uri

    def __iter__(self):
        request = urlopen(self.uri)
        for line in request.readlines():
            raw = json.loads(line)
            yield VirginiaTransformer(partner_id="va")._transform(raw)


jobpostings_va = VAJobposting(va_url)