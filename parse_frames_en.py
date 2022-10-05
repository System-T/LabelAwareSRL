import os
import xml.etree.ElementTree as ET
import pickle

frames_paths_1 = ['data/conll09/2009_conll_p2/data/CoNLL2009-ST-English/nb_frames', 'data/conll09/2009_conll_p2/data/CoNLL2009-ST-English/pb_frames']
frames_paths_2 = ['data/nombank.1.0/frames', 'data/propbank-frames/frames']
frames_paths_group = [frames_paths_1, frames_paths_2]

class FrameSet:
  def __init__(self):
    self.name = ''
    self.note = ''
    self.predicates = []

class Predicate:
  def __init__(self):
    self.note = ''
    self.lemma = ''
    self.rolesets = []

class RoleSet:
  def __init__(self):
    self.id = ''
    self.name = ''
    self.roles = {}
    self.examples = []

class Example:
  def __init__(self):
    self.name = ''
    self.text = ''
    self.predicate = ''
    self.arguments = []

def parse_and_save():
  for suffix, frames_paths in zip(['','_'], frames_paths_group):
    for pos, frames_path in zip(['noun','verb'], frames_paths):
      framesets = []
      roleset_constraints = {}
      predicate_role_definition = {}
      for fname in os.listdir(frames_path):
        if fname[-4:] == '.xml':
          root = ET.parse(os.path.join(frames_path, fname)).getroot()
          frameset = FrameSet()
          frameset.name = fname[:-4]
          for child1 in root:
            if child1.tag == 'note':
              frameset.note = child1.text
            elif child1.tag == 'predicate':
              predicate = Predicate()
              predicate.lemma = child1.attrib['lemma']
              for child2 in child1:
                if child2.tag == 'note':
                  predicate.note = child2.text
                elif child2.tag == 'roleset':
                  roleset = RoleSet()
                  roleset.name = child2.attrib['name']
                  roleset.id = child2.attrib['id']
                  roleset_constraints[roleset.id] = []
                  predicate_role_definition[roleset.id] = []
                  for child3 in child2:
                    if child3.tag == 'note':
                      roleset.note = child3.text
                    elif child3.tag == 'roles':
                      for child4 in child3:
                        if child4.tag == 'role':
                          roleset.roles[child4.attrib['n']] = child4.attrib['descr']
                          roleset_constraints[roleset.id].append("A"+child4.attrib['n'])
                          predicate_role_definition[roleset.id].append(("A"+child4.attrib['n'], child4.attrib['descr']))
                    elif child3.tag == 'example':
                      example = Example()
                      try:
                        example.name = child3.attrib['name']
                      except KeyError:
                        example.name = ''
                      for child4 in child3:
                        if child4.tag == 'text':
                          example.text = child4.text
                        elif child4.tag == 'arg':
                          example.arguments.append((child4.attrib['n'], child4.text))
                        elif child4.tag == 'rel':
                          example.predicate = child4.text
                      roleset.examples.append(example)
                  predicate.rolesets.append(roleset)
              frameset.predicates.append(predicate)
          framesets.append(frameset)

      #with open(f'pickles/roleset_constraints_{pos}{suffix}.pkl','wb') as f:
      #  pickle.dump(roleset_constraints, f)

      with open(f'pickles/predicate_role_definition_{pos}{suffix}.pkl','wb') as f:
        pickle.dump(predicate_role_definition, f)

      #with open(f'pickles/frame_sets_{pos}{suffix}.pkl','wb') as f:
      #  pickle.dump(framesets, f)

def get_examples():
  example_lines = []
  for pos in ['noun','verb']:
    with open(f'pickles/frame_sets_{pos}.pkl','rb') as f:
      framesets = pickle.load(f)
      for frameset in framesets:
        for predicate in frameset.predicates:
          for roleset in predicate.rolesets:
            for example in roleset.examples:
              pass


parse_and_save()
#load()