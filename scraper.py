import json
import pandas as pd
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
from pygments import highlight
from pygments.lexers import JsonLexer
from pygments.formatters import TerminalFormatter
import urllib.request
from google_play_scraper import Sort,reviews,app

sns.set(style='whitegrid', palette='muted', font_scale=1.2)

# Education Apps - Get this from Playstore url itself
app_packages = [
  'org.coursera.android',
  'org.edx.mobile',
  'com.mobile.simplilearn',
  'com.datacamp',
  'com.duolingo',
  'free.programming.programming',
  'air.nn.mobile.app.main',
  'com.microsoft.math',
  'com.zerodha.varsity',
  'justimaginestudio.com.mindset_achieveyourgoals',
  'com.deepstash',
  'co.brainly',
  'com.byjus.thelearningapp',
  'com.hrd.vocabulary',
  'com.memorado.brain.games'
]

#Get data of all the apps via google play scraper library
app_infos = []

for ap in tqdm(app_packages):
  info = app(ap, lang='en', country='in')
  del info['comments']
  app_infos.append(info)

#Dump to JSON
def print_json(json_object):
  json_str = json.dumps(
    json_object,
    indent=2,
    sort_keys=True,
    default=str
  )
  print(highlight(json_str, JsonLexer(), TerminalFormatter()))

#Sample Data
print_json(app_infos[0])


#Function to get the proper format of the title
#Example : DataCamp: Learn Python, SQL & R coding needs to be just DataCamp

def format_title(title:str):
    sep_index=title.find(':') if title.find(':') !=-1 else title.find('-')
    if sep_index!=-1:
        title=title[:sep_index]
    return title[:10]

fig, axs = plt.subplots(2, len(app_infos) // 2, figsize=(14, 5))


#Show some icons from the scraped data
for i, ax in enumerate(axs.flat):
  ai = app_infos[i]
  urllib.request.urlretrieve(ai['icon'],ai['title']+".png")
  pil_im = Image.open(ai['title']+".png", 'r')
  ax.imshow(np.asarray(pil_im))
 # img = plt.imread(ai['icon'])
  #ax.imshow(img)
  ax.set_title(format_title(ai['title']))
  ax.axis('off')

plt.show()

#Export the data to a dataframe
app_infos_df = pd.DataFrame(app_infos)
app_infos_df.to_csv('apps.csv', index=None, header=True)
#print(app_infos_df.head())


#Downloading reviews for different review scores and the most relevant and newest ones.
app_reviews = []

for ap in tqdm(app_packages):
  for score in list(range(1, 6)):
    for sort_order in [Sort.MOST_RELEVANT, Sort.NEWEST]:
      rvs, _ = reviews(
        ap,
        lang='en',
        country='in',
        sort=sort_order,
        count= 200 if score == 3 else 100, # Idea is to create 3 categories - Negative / Neutral / Postive for sentiment analysis.More reviews are downloaded if neutral else there might be an imbalanced dataset problem
        filter_score_with=score
      )
      for r in rvs:
        r['sortOrder'] = 'most_relevant' if sort_order == Sort.MOST_RELEVANT else 'newest'
        r['appId'] = ap
      app_reviews.extend(rvs)

print_json(app_reviews[1])
print(len(app_reviews))

app_reviews_df = pd.DataFrame(app_reviews)
app_reviews_df.to_csv('reviews.csv', index=None, header=True)