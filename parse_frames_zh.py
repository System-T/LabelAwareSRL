import os
import xml
import xml.etree.ElementTree as ET
import pickle

frames_path = 'data/conll09/2009_conll_p2/data/CoNLL2009-ST-Chinese/frames/'

class FrameSet:
  def __init__(self):
    self.id = ''
    self.roles = {}

class Example:
  def __init__(self):
    self.name = ''
    self.text = ''
    self.predicate = ''
    self.arguments = []

def parse_and_save():
  framesets = []
  predicate_role_definition = {}
  for fname in os.listdir(frames_path):
    if fname[-4:] == '.xml':
      try:
        root = ET.parse(os.path.join(frames_path, fname)).getroot()
      except xml.etree.ElementTree.ParseError:
        print(f'{fname} is malformed')
      for child1 in root:
        if child1.tag == 'id':
          name = child1.text
        elif child1.tag == 'frameset':
          frameset = FrameSet()
          #print(child1.attrib['id'][1])
          frameset.id = name.strip() + '.0' + child1.attrib['id'][1]
          predicate_role_definition[frameset.id] = []
          for child2 in child1:
            if child2.tag == 'role':
              frameset.roles[child2.attrib['argnum']] = child2.attrib['argrole']
              predicate_role_definition[frameset.id].append(("A"+child2.attrib['argnum'], child2.attrib['argrole']))
          framesets.append(frameset)

    with open(f'pickles/zh_predicate_role_definition.pkl','wb') as f:
      pickle.dump(predicate_role_definition, f)

    with open(f'pickles/zh_frame_sets.pkl','wb') as f:
      pickle.dump(framesets, f)

def load():
  with open(f'pickles/zh_predicate_role_definition.pkl','rb') as f:
    framesets = pickle.load(f)
    print(framesets)

parse_and_save()
#load()